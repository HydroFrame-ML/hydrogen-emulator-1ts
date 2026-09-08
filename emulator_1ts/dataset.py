import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from parflow.tools.io import read_pfb, read_pfb_sequence
from torch.utils.data import Dataset

from .logger import info, verbose


def timestep_files(run_name: str, member_dir: Path, variable: str) -> Dict[int, Path]:
    """Map timestep number to PFB path for one output variable.

    Module level so anything that has to see the same files as training 
    the scaler fitter, for example, resolves them by the same rule.
    """

    pattern = re.compile(
        rf"^{re.escape(run_name)}\.out\.{re.escape(variable)}\.(\d+)\.pfb$"
    )
    result = {}
    for file_path in member_dir.glob(f"{run_name}.out.{variable}.*.pfb"):
        match = pattern.match(file_path.name)
        if match:
            timestep = int(match.group(1))
            if timestep in result:
                raise ValueError(
                    f"Duplicate {variable} timestep {timestep} in {member_dir}"
                )
            result[timestep] = file_path
    return result


def resolve_parameter_file(run_name: str, member_dir: Path, parameter: str) -> Path:
    output_file = member_dir / f"{run_name}.out.{parameter}.pfb"
    input_file = member_dir / f"{parameter}.pfb"
    # MJB contains two masks: ``mask.pfb`` is the one-channel binary domain
    # mask, while ``mjb.out.mask.pfb`` is a ten-layer diagnostic encoded as
    # 0/99999. Preserve the historical output-first order for other fields.
    candidates = (
        (input_file, output_file)
        if parameter == "mask"
        else (output_file, input_file)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"No static parameter file found for {parameter!r} in {member_dir}; "
        f"tried {[str(path) for path in candidates]}"
    )


def normalize_member_id(member_id) -> str:
    member_id = str(member_id)
    if member_id.startswith("member_"):
        member_id = member_id[len("member_") :]
    return member_id


def metadata_member_ids(base_dir: Path) -> List[str]:
    """Member ids listed in ``metadata.csv``, or an empty list if absent."""

    metadata_path = Path(base_dir) / "metadata.csv"
    if not metadata_path.is_file():
        return []

    with metadata_path.open(newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if not reader.fieldnames or "member_id" not in reader.fieldnames:
            raise ValueError(f"metadata.csv has no member_id column: {metadata_path}")
        return [normalize_member_id(row["member_id"]) for row in reader]


def discover_member_directories(base_dir: Path, member_ids=None):
    """Resolve ``(member_id, path)`` pairs for an ensemble or a single run.

    A directory with no ``member_*`` children is treated as one unnamed run, so
    callers handle both layouts without branching. Paths are resolved locally
    rather than read from ``metadata.csv``, whose stored directories may be
    stale; the file is consulted only for which members exist.
    """

    base_dir = Path(base_dir)
    discovered = {
        path.name[len("member_") :]: path
        for path in sorted(base_dir.glob("member_*"))
        if path.is_dir()
    }

    if not discovered:
        if member_ids:
            raise ValueError(
                f"member_ids were supplied, but no member_* directories exist in {base_dir}"
            )
        return [("single", base_dir)]

    requested = (
        [normalize_member_id(member_id) for member_id in member_ids]
        if member_ids is not None
        else metadata_member_ids(base_dir) or list(discovered)
    )
    if not requested:
        raise ValueError(f"No ensemble members selected from {base_dir}")

    missing = [member_id for member_id in requested if member_id not in discovered]
    if missing:
        raise ValueError(
            f"Requested ensemble members are missing from {base_dir}: {missing}"
        )
    return [(member_id, discovered[member_id]) for member_id in requested]


@dataclass(frozen=True)
class SampleIndex:
    """Location of one spatial and temporal sample within one ensemble member."""

    member_id: str
    start_timestep: int
    x_start: int
    y_start: int


@dataclass(frozen=True)
class MemberFiles:
    """Files and valid temporal windows belonging to one ensemble member."""

    member_id: str
    directory: Path
    pressure_files: Mapping[int, Path]
    evaptrans_files: Mapping[int, Path]
    parameter_files: Mapping[str, Path]
    valid_start_timesteps: Tuple[int, ...]


class ParFlowDataset(Dataset):
    """ParFlow pressure-transition dataset.

    ``data_location`` may be either a traditional single-run directory or an
    ensemble root containing ``member_*`` directories. In ensemble mode every
    temporal window is validated and indexed within one member, so a multistep
    sample can never roll over into the next member.
    """

    def __init__(
        self,
        data_location,
        run_name,
        parameters,
        patch_size_x,
        patch_size_y,
        overlap_x,
        overlap_y,
        n_evaptrans=0,
        n_timesteps=1,
        shuffle=False,
        dtype=torch.float64,
        preload=True,
        cache_size=64,
        member_ids=None,
        **kwargs,
    ):
        super().__init__()
        self.base_dir = Path(data_location)
        self.run_name = run_name
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.n_evaptrans = n_evaptrans
        self.n_timesteps = n_timesteps
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.shuffle = shuffle
        self.dtype = dtype
        self.preload = preload
        self.cache_size = cache_size

        if self.n_timesteps < 1:
            raise ValueError("n_timesteps must be at least 1")

        # Split the parameter list from the number of selected layers.
        try:
            params, layers = zip(*parameters)
        except ValueError as exc:
            raise ValueError(
                "parameters must contain (parameter_name, number_of_layers) pairs"
            ) from exc
        self.parameter_list = list(params)
        self.param_nlayer = list(layers)

        # Static fields are member-specific. Dynamic cache entries are also
        # naturally member-specific because their absolute paths include the
        # member directory.
        self.static_data_dict: Dict[Tuple[str, str], np.ndarray] = {}
        self.cache: Dict[str, np.ndarray] = {}

        member_directories = self._discover_member_directories(member_ids)
        self.members: Dict[str, MemberFiles] = {}
        for member_id, member_dir in member_directories:
            member = self._build_member_index(member_id, member_dir)
            if not member.valid_start_timesteps:
                raise ValueError(
                    f"Member {member_id} has no complete {self.n_timesteps}-step windows"
                )
            self.members[member_id] = member

        first_member = next(iter(self.members.values()))
        first_pressure = first_member.pressure_files[
            min(first_member.pressure_files)
        ]
        size_test = read_pfb(str(first_pressure))
        self.X_EXTENT = size_test.shape[2]
        self.Y_EXTENT = size_test.shape[1]
        self.Z_EXTENT = size_test.shape[0]

        x_starts = self._patch_starts(
            self.X_EXTENT, self.patch_size_x, self.overlap_x, "x"
        )
        y_starts = self._patch_starts(
            self.Y_EXTENT, self.patch_size_y, self.overlap_y, "y"
        )

        self.sample_index: List[SampleIndex] = []
        for member in self.members.values():
            for start_timestep in member.valid_start_timesteps:
                for y_start in y_starts:
                    for x_start in x_starts:
                        self.sample_index.append(
                            SampleIndex(
                                member_id=member.member_id,
                                start_timestep=start_timestep,
                                x_start=x_start,
                                y_start=y_start,
                            )
                        )

        # T_EXTENT is retained for callers that inspect it. Unlike the former
        # global xbatcher axis, it is the total number of member-local temporal
        # windows and is not used to construct samples.
        self.T_EXTENT = sum(
            len(member.valid_start_timesteps) for member in self.members.values()
        )

        info(
            f"Found {len(self.members)} member(s), {self.T_EXTENT} temporal "
            f"windows, and {len(self.sample_index)} total samples"
        )
        verbose(
            f"Dataset dimensions: Z={self.Z_EXTENT}, Y={self.Y_EXTENT}, "
            f"X={self.X_EXTENT}; horizon={self.n_timesteps}"
        )

        self.generate_namelist()

        if self.preload:
            self._preload_static_parameters()

    @staticmethod
    def _normalize_member_id(member_id) -> str:
        return normalize_member_id(member_id)

    def _metadata_member_ids(self) -> List[str]:
        return metadata_member_ids(self.base_dir)

    def _discover_member_directories(self, member_ids) -> List[Tuple[str, Path]]:
        """Resolve member paths locally instead of trusting stored absolute paths."""

        return discover_member_directories(self.base_dir, member_ids)

    def _timestep_files(self, member_dir: Path, variable: str) -> Dict[int, Path]:
        return timestep_files(self.run_name, member_dir, variable)

    def _resolve_parameter_file(self, member_dir: Path, parameter: str) -> Path:
        return resolve_parameter_file(self.run_name, member_dir, parameter)

    def _build_member_index(self, member_id: str, member_dir: Path) -> MemberFiles:
        pressure_files = self._timestep_files(member_dir, "press")
        evaptrans_files = self._timestep_files(member_dir, "evaptrans")
        if not pressure_files:
            raise ValueError(
                f"No pressure files found for run {self.run_name} in member {member_id}"
            )
        if not evaptrans_files:
            raise ValueError(
                f"No evaptrans files found for run {self.run_name} in member {member_id}"
            )

        parameter_files = {
            parameter: self._resolve_parameter_file(member_dir, parameter)
            for parameter in self.parameter_list
        }

        # A start t is valid only if p[t], p[t+1:t+H], and ET[t+1:t+H]
        # all exist in this same member. Explicit membership checks also handle
        # incomplete runs and non-zero/non-contiguous timestep numbering.
        valid_starts = []
        for start_timestep in sorted(pressure_files):
            future_timesteps = range(
                start_timestep + 1, start_timestep + self.n_timesteps + 1
            )
            if all(
                timestep in pressure_files and timestep in evaptrans_files
                for timestep in future_timesteps
            ):
                valid_starts.append(start_timestep)

        verbose(
            f"Member {member_id}: {len(pressure_files)} pressure files, "
            f"{len(evaptrans_files)} evaptrans files, "
            f"{len(valid_starts)} valid windows"
        )
        return MemberFiles(
            member_id=member_id,
            directory=member_dir,
            pressure_files=pressure_files,
            evaptrans_files=evaptrans_files,
            parameter_files=parameter_files,
            valid_start_timesteps=tuple(valid_starts),
        )

    @staticmethod
    def _patch_starts(extent: int, patch_size: int, overlap: int, axis: str) -> List[int]:
        if patch_size < 1 or patch_size > extent:
            raise ValueError(
                f"patch_size_{axis} must be between 1 and the {axis} extent ({extent})"
            )
        if overlap < 0 or overlap >= patch_size:
            raise ValueError(
                f"overlap_{axis} must be non-negative and smaller than patch_size_{axis}"
            )
        stride = patch_size - overlap
        # Match xbatcher's previous return_partial=False behavior.
        return list(range(0, extent - patch_size + 1, stride))

    def generate_namelist(self):
        """Record the exact channel ordering used by the model."""

        self.PRESSURE_NAMES = [f"pressure_{i}" for i in range(self.Z_EXTENT)]
        evaptrans_layers = self._selected_layer_indices(
            self.Z_EXTENT, self.n_evaptrans
        )
        self.EVAPTRANS_NAMES = [f"evaptrans_{i}" for i in evaptrans_layers]
        self.PARAM_NAMES = []
        self.OUTPUT_NAMES = [f"pressure_{i}" for i in range(self.Z_EXTENT)]

        first_member = next(iter(self.members.values()))
        patch_keys = {
            "x": {"start": 0, "stop": min(2, self.X_EXTENT)},
            "y": {"start": 0, "stop": min(2, self.Y_EXTENT)},
        }

        for parameter, n_lay in zip(self.parameter_list, self.param_nlayer):
            param_temp = read_pfb(
                str(first_member.parameter_files[parameter]), keys=patch_keys
            )
            selected_layers = self._selected_layer_indices(param_temp.shape[0], n_lay)
            if param_temp.shape[0] == 1:
                self.PARAM_NAMES.append(parameter)
            else:
                self.PARAM_NAMES.extend(
                    f"{parameter}_{layer}" for layer in selected_layers
                )

    @staticmethod
    def _selected_layer_indices(n_layers: int, selection: int) -> List[int]:
        if selection > 0:
            return list(range(min(selection, n_layers)))
        if selection < 0:
            start = max(0, n_layers + selection)
            return list(range(start, n_layers))

        return list(range(n_layers))

    @classmethod
    def _select_layers(cls, data: np.ndarray, selection: int) -> np.ndarray:
        return data[cls._selected_layer_indices(data.shape[0], selection), :, :]

    def _preload_static_parameters(self):
        info("Preloading static parameters for each ensemble member...")
        for member in self.members.values():
            for parameter in self.parameter_list:
                cache_key = (member.member_id, parameter)
                self.static_data_dict[cache_key] = read_pfb(
                    str(member.parameter_files[parameter])
                )
        verbose("Static parameters preloaded successfully.")

    def _read_cached_pfb(
        self, file_path, x_min=None, x_max=None, y_min=None, y_max=None
    ):
        """Read and cache a complete PFB, then optionally extract a patch."""

        file_path = str(file_path)
        if file_path in self.cache:
            data = self.cache[file_path]
        else:
            data = read_pfb(file_path)
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[file_path] = data

        if x_min is not None:
            return data[:, y_min : y_max + 1, x_min : x_max + 1]
        return data

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        sample = self.sample_index[idx]
        member = self.members[sample.member_id]
        x_min = sample.x_start
        x_max = x_min + self.patch_size_x - 1
        y_min = sample.y_start
        y_max = y_min + self.patch_size_y - 1

        patch_keys = {
            "x": {"start": x_min, "stop": x_max + 1},
            "y": {"start": y_min, "stop": y_max + 1},
        }
        target_timesteps = list(
            range(
                sample.start_timestep + 1,
                sample.start_timestep + self.n_timesteps + 1,
            )
        )

        # These paths are selected from one MemberFiles object by construction.
        pressure_file_sequence = [
            member.pressure_files[sample.start_timestep],
            *(member.pressure_files[timestep] for timestep in target_timesteps),
        ]
        evaptrans_file_sequence = [
            member.evaptrans_files[timestep] for timestep in target_timesteps
        ]

        pressure_sequence = read_pfb_sequence(
            [str(path) for path in pressure_file_sequence], keys=patch_keys
        )
        pressure_valid = np.isfinite(pressure_sequence) & (pressure_sequence >= -9999)
        state_data = np.where(pressure_valid[0], pressure_sequence[0], 0.0)
        target_sequence = np.where(
            pressure_valid[1:], pressure_sequence[1:], 0.0
        )
        target_valid = pressure_valid[1:]

        evaptrans_sequence_raw = read_pfb_sequence(
            [str(path) for path in evaptrans_file_sequence], keys=patch_keys
        )
        evaptrans_sequence = np.stack(
            [
                self._select_layers(evaptrans, self.n_evaptrans)
                for evaptrans in evaptrans_sequence_raw
            ]
        )

        parameter_data = []
        for parameter, n_lay in zip(self.parameter_list, self.param_nlayer):
            if self.preload:
                param_temp = self.static_data_dict[(member.member_id, parameter)]
                param_temp = param_temp[:, y_min : y_max + 1, x_min : x_max + 1]
            else:
                param_temp = read_pfb(
                    str(member.parameter_files[parameter]), keys=patch_keys
                )
            parameter_data.append(self._select_layers(param_temp, n_lay))
        parameter_data = np.concatenate(parameter_data, axis=0)

        # A one-step item has [C,Y,X] targets/ET/mask; multistep items retain
        # an explicit leading time axis.
        state_tensor = torch.from_numpy(state_data).to(self.dtype)
        parameter_tensor = torch.from_numpy(parameter_data).to(self.dtype)
        target_tensor = torch.from_numpy(target_sequence).to(self.dtype)
        evaptrans_tensor = torch.from_numpy(evaptrans_sequence).to(self.dtype)
        target_mask = torch.from_numpy(target_valid).to(torch.bool)
        if self.n_timesteps == 1:
            target_tensor = target_tensor[0]
            evaptrans_tensor = evaptrans_tensor[0]
            target_mask = target_mask[0]

        # Pressure sentinels have already been replaced with neutral placeholders;
        # clean any invalid ET/static values before tensors reach the model. The
        # pressure mask is returned so placeholders never contribute to loss.
        for tensor in (
            state_tensor,
            parameter_tensor,
            target_tensor,
            evaptrans_tensor,
        ):
            tensor[~torch.isfinite(tensor)] = 0
            tensor[tensor < -9999] = 0

        return (
            state_tensor,
            evaptrans_tensor,
            parameter_tensor,
            target_tensor,
            target_mask,
        )

    def clear_cache(self):
        """Clear the dynamic PFB cache."""

        self.cache.clear()
