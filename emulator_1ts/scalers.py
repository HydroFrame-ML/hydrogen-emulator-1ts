"""Per-channel standardization statistics.

A scaler set maps a channel name to ``(mean, std)``. The models apply these as
plain z-scores, so every entry must describe the quantity the channel actually
carries.

Pressure keys are named ``pressure_{layer}``. They were historically named
``press_diff_{layer}``, which was misleading: the packaged CONUS2.1 statistics
under that key are moments of the *absolute* pressure field, and that is what
``scale_pressure`` is applied to. Two of the shipped files
(``*_original.yaml``, ``*_adjusted.yaml``) do hold genuine hourly-difference
moments under the old key, and their sigmas are ~5 orders of magnitude smaller,
so loading one of those in place of a pressure file silently destroys the
scaling. The legacy key is still accepted so existing scaler files keep
working, but it now warns.
"""

import os
import warnings
from pathlib import Path

import yaml

PRESSURE_KEY_PREFIX = 'pressure'
LEGACY_PRESSURE_KEY_PREFIX = 'press_diff'


def _entry_moments(name, entry, source):
    """Pull ``(mean, std)`` out of one scaler entry, or say why it cannot."""

    if isinstance(entry, dict):
        missing = [key for key in ('mean', 'std') if key not in entry]
        if missing:
            raise ValueError(
                f"Scaler {name!r} in {source} is missing {missing}"
            )
        raw_mean, raw_std = entry['mean'], entry['std']
    else:
        try:
            raw_mean, raw_std = entry
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Scaler {name!r} in {source} must be a (mean, std) pair or a "
                f"mapping with 'mean' and 'std'; got {entry!r}"
            ) from exc

    try:
        return float(raw_mean), float(raw_std)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Scaler {name!r} in {source} has non-numeric moments: "
            f"mean={raw_mean!r}, std={raw_std!r}"
        ) from exc


def validate_scalers(scalers, source='scalers'):
    """Return a checked ``{name: (mean, std)}`` dict.

    Scaling divides by sigma, so an unusable sigma has to be caught here rather
    than surfacing later as a NaN loss pointing at the wrong thing. A negative
    or non-finite moment is always corruption and raises. A sigma of exactly
    zero is different: it is what fitting produces for a field that is constant
    over the domain, which is entirely possible for a static like ``mannings``
    on a small basin. Following the StandardScaler convention these files name,
    that becomes a sigma of 1, so the channel centers to zeros and still
    unscales back to its constant. It warns, because a zero sigma can equally
    mean the fit ran over a single sample.

    That substitution is the same one the packaged ``*_adjusted*`` files apply
    by hand to the ``*_original*`` files they are otherwise identical to.
    """

    validated = {}
    constant_channels = []
    for name, entry in scalers.items():
        mean, std = _entry_moments(name, entry, source)
        if mean != mean or mean in (float('inf'), float('-inf')):
            raise ValueError(
                f"Scaler {name!r} in {source} has a non-finite mean: {mean}"
            )
        if std != std or std in (float('inf'), float('-inf')):
            raise ValueError(
                f"Scaler {name!r} in {source} has a non-finite std: {std}"
            )
        if std < 0.0:
            raise ValueError(
                f"Scaler {name!r} in {source} has a negative std: {std}"
            )
        if std == 0.0:
            constant_channels.append(name)
            std = 1.0
        validated[name] = (mean, std)

    if constant_channels:
        # One warning per load rather than per channel: the packaged
        # difference files trip this on 30+ statics at once.
        listed = ', '.join(constant_channels[:8])
        if len(constant_channels) > 8:
            listed += f", ... ({len(constant_channels)} total)"
        warnings.warn(
            f"{source}: std=0 for {listed}. Those channels are constant over "
            "the domain, so std=1 is used and they scale to zeros. Check that "
            "the fit covered more than one sample.",
            RuntimeWarning,
            stacklevel=3,
        )
    return validated


def create_scalers_from_yaml(file):
    with open(file, 'r') as f:
        lookup = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(lookup, dict):
        raise ValueError(
            f"Scaler file {file} must contain a mapping of channel name to "
            f"moments; got {type(lookup).__name__}"
        )
    return validate_scalers(lookup, source=str(file))


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCALER_PATH = f'{HERE}/CONUS21_scalers_parflow_names.yaml'
DEFAULT_SCALERS = create_scalers_from_yaml(DEFAULT_SCALER_PATH)


def find_scaler_file(path):
    """Resolve a scaler path, falling back to the files packaged here.

    This lets a config name a packaged set (``CONUS21_scalers_parflow_names.yaml``)
    without hard-coding an installation path, while still accepting an absolute
    or working-directory-relative path to a basin-specific file.
    """

    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    packaged = Path(HERE) / candidate.name
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(
        f"No scaler file found for {path!r}; tried {str(candidate)} and "
        f"{str(packaged)}"
    )


def load_scalers(scalers=None):
    """Normalize a scaler specification into a ``{name: (mean, std)}`` dict.

    Accepts an already-built mapping, a path to a YAML file, or ``None`` for the
    packaged CONUS2.1 statistics. Model constructors route their ``scalers``
    argument through this, so a config can select a basin-specific file with
    ``model_def.scalers: /path/to/scalers.yaml``.
    """

    if scalers is None:
        # Already validated when the module was imported.
        return dict(DEFAULT_SCALERS)
    if isinstance(scalers, (str, Path)):
        return create_scalers_from_yaml(find_scaler_file(scalers))
    if isinstance(scalers, dict):
        return validate_scalers(scalers, source='the supplied scalers')
    raise TypeError(
        "scalers must be a mapping, a path to a YAML file, or None; got "
        f"{type(scalers).__name__}"
    )


def resolve_pressure_scaler_names(scalers, pressure_names=None, n_layers=None):
    """Return the scaler key for each pressure layer, newest naming first.

    ``pressure_{i}`` is preferred and ``press_diff_{i}`` accepted with a
    warning, so the channel-to-key mapping is settled once at construction
    rather than rebuilt inside the scaling loops.

    A scaler set that covers no pressure layers yields an empty list rather
    than an error: some sets legitimately describe statics only, and such a
    model simply cannot scale pressure.
    """

    if n_layers is None:
        if pressure_names is not None:
            n_layers = len(pressure_names)
        else:
            n_layers = 0
            while (
                f'{PRESSURE_KEY_PREFIX}_{n_layers}' in scalers
                or f'{LEGACY_PRESSURE_KEY_PREFIX}_{n_layers}' in scalers
            ):
                n_layers += 1

    names = []
    legacy_used = False
    for index in range(n_layers):
        key = f'{PRESSURE_KEY_PREFIX}_{index}'
        legacy_key = f'{LEGACY_PRESSURE_KEY_PREFIX}_{index}'
        if key in scalers:
            names.append(key)
        elif legacy_key in scalers:
            names.append(legacy_key)
            legacy_used = True
        else:
            raise ValueError(
                f"Scalers contain no entry for pressure layer {index} "
                f"(looked for {key!r} and {legacy_key!r})"
            )

    if legacy_used:
        warnings.warn(
            "Scaler keys 'press_diff_*' are deprecated; rename them to "
            "'pressure_*'. Pressure scalers are applied to absolute pressures, "
            "so confirm these entries are pressure moments and not "
            "pressure-difference moments.",
            FutureWarning,
            stacklevel=2,
        )
    return names
