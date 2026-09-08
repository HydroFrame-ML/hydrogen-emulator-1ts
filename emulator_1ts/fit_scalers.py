"""Fit standardization statistics from ParFlow output.

The packaged scalers describe CONUS2.1 over WY2003. Training on a subset basin
or on an ensemble that perturbs the statics standardizes with the wrong
distribution unless a matching scaler set is fitted first. This module produces
one, from a single run directory or from an ensemble of ``member_*`` runs, and
writes it in the YAML shape ``emulator_1ts.scalers`` loads.

Two things it does deliberately:

* Statistics cover **active cells only**. Inactive basin cells are constant
  placeholders, so including them shrinks every sigma and biases every mean
  toward the placeholder value. The domain mask is used when one is present,
  and ParFlow sentinels (non-finite, or below ``-9999``) are dropped either
  way, matching the rule the dataset uses when it builds tensors.
* Moments are accumulated with Welford's algorithm in float64 rather than
  summing squares. A constant field then yields exactly zero variance instead
  of a small negative number whose square root is NaN, which matters here
  because several CONUS statics -- ``ssat``, ``vg_n``, ``pf_flowbarrier`` --
  really are constant.

Fit on the **training** members only. Fitting on validation or test members
leaks their distribution into the model's inputs.

Usage::

    python -m emulator_1ts.fit_scalers --config my_config.yaml --output mjb.yaml
    python -m emulator_1ts.fit_scalers --data-location runs/mjb --run-name mjb \\
        --parameters perm_x perm_y porosity mask --output mjb.yaml
"""

import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from parflow.tools.io import read_pfb

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from emulator_1ts.dataset import (
        discover_member_directories,
        resolve_parameter_file,
        timestep_files,
    )
    from emulator_1ts.logger import info, set_log_level, verbose
    from emulator_1ts.scalers import validate_scalers
else:
    from .dataset import (
        discover_member_directories,
        resolve_parameter_file,
        timestep_files,
    )
    from .logger import info, set_log_level, verbose
    from .scalers import validate_scalers


SENTINEL_FLOOR = -9999.0


class RunningMoments:
    """Streaming mean and population variance (Welford, chunked merge).

    Chunked so each PFB layer is folded in as one array operation, and stable
    so a constant field returns std exactly 0 rather than the small negative
    variance a sum-of-squares accumulator can produce.
    """

    __slots__ = ('count', 'mean', 'm2')

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray):
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        chunk_count = values.size
        chunk_mean = float(values.mean())
        chunk_m2 = float(((values - chunk_mean) ** 2).sum())

        if self.count == 0:
            self.count, self.mean, self.m2 = chunk_count, chunk_mean, chunk_m2
            return

        total = self.count + chunk_count
        delta = chunk_mean - self.mean
        self.mean += delta * chunk_count / total
        self.m2 += chunk_m2 + delta * delta * self.count * chunk_count / total
        self.count = total

    @property
    def std(self) -> float:
        if self.count == 0:
            return 0.0
        # Population std, matching CONUS2_Data_Prep/pressure_scalers.py.
        return float(np.sqrt(max(self.m2, 0.0) / self.count))

    def as_entry(self) -> dict:
        return {
            'mean': float(self.mean),
            'std': self.std,
            'type': 'StandardScaler',
        }


def _active_values(layer: np.ndarray, mask_layer):
    """Finite, non-sentinel values of one layer, restricted to active cells."""

    # Same rule the dataset applies when it builds tensors, so the fit covers
    # exactly the values the model will be fed.
    valid = np.isfinite(layer) & (layer >= SENTINEL_FLOOR)
    if mask_layer is not None:
        valid &= mask_layer
    return layer[valid]


def _read_mask(run_name: str, member_dir: Path):
    """Boolean active-cell mask as ``(z, y, x)``, or None when absent.

    A one-layer mask is left with a leading axis of 1 so it broadcasts over any
    number of layers; MJB's ten-layer 0/99999 diagnostic is used per layer.
    """

    try:
        mask_path = resolve_parameter_file(run_name, member_dir, 'mask')
    except ValueError:
        verbose(f"No mask found in {member_dir}; using sentinel filtering only")
        return None
    mask = np.asarray(read_pfb(str(mask_path)), dtype=np.float64)
    return mask > 0


def _mask_layer(mask, index: int):
    if mask is None:
        return None
    if mask.shape[0] == 1:
        return mask[0]
    return mask[index]


def _static_channel_names(parameter: str, n_layers: int):
    """Channel names for one static, matching ``ParFlowDataset.PARAM_NAMES``."""

    if n_layers == 1:
        return [parameter]
    return [f'{parameter}_{layer}' for layer in range(n_layers)]


def fit_scalers(
    data_location,
    run_name,
    parameters=None,
    member_ids=None,
    timestep_interval=1,
    max_timesteps=None,
):
    """Fit ``{channel: {mean, std, type}}`` over a run or an ensemble.

    Args:
        data_location: A run directory, or an ensemble root of ``member_*``
            directories.
        run_name: PFB prefix, as in ``{run_name}.out.press.00000.pfb``.
        parameters: Static parameter names to fit. Names come out matching the
            dataset's channel naming, so multi-layer statics become
            ``{parameter}_{layer}``. ``mask`` is accepted and skipped, since it
            is never scaled.
        member_ids: Ensemble members to include; defaults to ``metadata.csv``
            when present, otherwise every discovered member. Pass the training
            members only.
        timestep_interval: Stride over the available timesteps. The packaged
            CONUS statistics used a prime stride to avoid sampling the same
            hour of day repeatedly; the same reasoning applies to any run long
            enough to carry a diurnal cycle.
        max_timesteps: Optional cap on timesteps per member, for a quick fit.

    Returns:
        The fitted scaler dict, already validated.
    """

    base_dir = Path(data_location)
    if not base_dir.is_dir():
        raise ValueError(f"data_location is not a directory: {base_dir}")
    if timestep_interval < 1:
        raise ValueError("timestep_interval must be at least 1")

    parameters = list(parameters or [])
    members = discover_member_directories(base_dir, member_ids)
    info(
        f"Fitting scalers over {len(members)} member(s) in {base_dir} "
        f"(run_name={run_name})"
    )

    pressure_moments = {}
    evaptrans_moments = {}
    static_moments = {}
    total_pressure_steps = 0
    total_evaptrans_steps = 0

    for member_id, member_dir in members:
        mask = _read_mask(run_name, member_dir)
        pressure_files = timestep_files(run_name, member_dir, 'press')
        evaptrans_files = timestep_files(run_name, member_dir, 'evaptrans')
        if not pressure_files:
            raise ValueError(
                f"No pressure files for run {run_name!r} in {member_dir}"
            )

        for variable, files, moments in (
            ('pressure', pressure_files, pressure_moments),
            ('evaptrans', evaptrans_files, evaptrans_moments),
        ):
            selected = sorted(files)[::timestep_interval]
            if max_timesteps is not None:
                selected = selected[:max_timesteps]
            for timestep in selected:
                data = np.asarray(read_pfb(str(files[timestep])), dtype=np.float64)
                for layer in range(data.shape[0]):
                    name = f'{variable}_{layer}'
                    moments.setdefault(name, RunningMoments()).update(
                        _active_values(data[layer], _mask_layer(mask, layer))
                    )
            if variable == 'pressure':
                total_pressure_steps += len(selected)
            else:
                total_evaptrans_steps += len(selected)
            verbose(
                f"Member {member_id}: {len(selected)} {variable} timestep(s)"
            )

        for parameter in parameters:
            if parameter == 'mask':
                continue
            path = resolve_parameter_file(run_name, member_dir, parameter)
            data = np.asarray(read_pfb(str(path)), dtype=np.float64)
            names = _static_channel_names(parameter, data.shape[0])
            for layer, name in enumerate(names):
                static_moments.setdefault(name, RunningMoments()).update(
                    _active_values(data[layer], _mask_layer(mask, layer))
                )

    if not evaptrans_moments:
        info("No evaptrans files found; the fitted set covers pressure and statics only")

    info(
        f"Fitted {len(pressure_moments)} pressure, {len(evaptrans_moments)} "
        f"evaptrans, and {len(static_moments)} static channel(s) from "
        f"{total_pressure_steps} pressure and {total_evaptrans_steps} "
        f"evaptrans timestep(s)"
    )

    scalers = {}
    for moments in (pressure_moments, evaptrans_moments, static_moments):
        for name, running in moments.items():
            scalers[name] = running.as_entry()

    # Round-trip through the loader's validation so a fitted file can never be
    # written in a shape the models would reject, and so constant channels are
    # reported here rather than at the start of training.
    validate_scalers(
        {name: (entry['mean'], entry['std']) for name, entry in scalers.items()},
        source=f'scalers fitted from {base_dir}',
    )
    return scalers


def write_scalers(scalers, output_path, provenance=None):
    """Write a fitted scaler set, with a provenance header."""

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Fitted by emulator_1ts.fit_scalers',
        f'# Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}',
    ]
    for key, value in (provenance or {}).items():
        lines.append(f'# {key}: {value}')
    header = '\n'.join(lines) + '\n'

    with output_path.open('w') as handle:
        handle.write(header)
        yaml.safe_dump(scalers, handle, sort_keys=False)
    info(f"Wrote {len(scalers)} scaler entries to {output_path}")
    return output_path


def settings_from_config(config):
    """Pull fitting inputs out of a training config.

    Fitting from the config that will train guarantees the two agree on the
    run name, the statics, and which members are training members.
    """

    data_def = config.get('data_def', {})
    data_location = data_def.get('train_data_location')
    if not data_location:
        raise ValueError("Config has no data_def.train_data_location")

    parameters = []
    for entry in data_def.get('parameters', []):
        parameters.append(entry[0] if isinstance(entry, (list, tuple)) else entry)

    return {
        'data_location': data_location,
        'run_name': data_def.get('run_name', config.get('name')),
        'parameters': parameters,
        'member_ids': data_def.get('train_member_ids'),
    }


def main(argv=None):
    parser = ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '--config',
        help="Training config to take data location, run name, statics, and "
             "training member ids from",
    )
    parser.add_argument('--data-location', help="Run directory or ensemble root")
    parser.add_argument('--run-name', help="PFB prefix, e.g. 'mjb'")
    parser.add_argument(
        '--parameters', nargs='*', help="Static parameter names to fit"
    )
    parser.add_argument(
        '--member-ids',
        nargs='*',
        help="Ensemble members to fit over; defaults to metadata.csv or all members",
    )
    parser.add_argument('--output', required=True, help="Scaler YAML to write")
    parser.add_argument(
        '--timestep-interval',
        type=int,
        default=1,
        help="Stride over timesteps; a prime stride avoids diurnal aliasing",
    )
    parser.add_argument(
        '--max-timesteps', type=int, help="Cap on timesteps per member"
    )
    parser.add_argument(
        '--log-level',
        choices=['silent', 'info', 'verbose'],
        default='info',
    )
    args = parser.parse_args(argv)
    set_log_level(args.log_level)

    settings = {
        'data_location': args.data_location,
        'run_name': args.run_name,
        'parameters': args.parameters,
        'member_ids': args.member_ids,
    }
    if args.config:
        with open(args.config) as handle:
            config = yaml.safe_load(handle)
        from_config = settings_from_config(config)
        # Explicit flags win over the config they are paired with.
        settings = {
            key: value if value is not None else from_config[key]
            for key, value in settings.items()
        }
    if not settings['data_location']:
        parser.error("--data-location is required without --config")
    if not settings['run_name']:
        parser.error("--run-name is required without --config")

    scalers = fit_scalers(
        timestep_interval=args.timestep_interval,
        max_timesteps=args.max_timesteps,
        **settings,
    )
    write_scalers(
        scalers,
        args.output,
        provenance={
            'data_location': settings['data_location'],
            'run_name': settings['run_name'],
            'members': settings['member_ids'] or 'all discovered',
            'timestep_interval': args.timestep_interval,
        },
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
