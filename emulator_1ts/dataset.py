import xarray as xr
import os
import torch
import xbatcher as xb
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from parflow.tools.io import read_pfb, read_pfb_sequence
from torch.utils.data import Dataset
from functools import lru_cache
import concurrent.futures

from .logger import info, verbose, error

class ParFlowDataset(Dataset):

    def __init__(
        self, data_location, run_name,
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
        cache_size=64, **kwargs,
    ):
        super().__init__()
        self.base_dir = data_location
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

        #Split the parameter_list from param_nlayer
        params, layers = zip(*parameters)
        self.parameter_list = list(params)
        self.param_nlayer = list(layers)

        # NOTE: Used for internal debugging, should be set to False to disable

        # Cache for frequently accessed PFB files
        self.cache = {}
        self.cache_size = cache_size

        # Find and organize data files
        self.pressure_files = sorted(glob(f'{self.base_dir}/{run_name}.out.press.*.pfb'))
        self.evaptrans_files = sorted(glob(f'{self.base_dir}/{run_name}.out.evaptrans.*.pfb'))
        self.velx_files = sorted(glob(f'{self.base_dir}/{run_name}.out.velx.*.pfb'))
        self.vely_files = sorted(glob(f'{self.base_dir}/{run_name}.out.vely.*.pfb'))
        self.velz_files = sorted(glob(f'{self.base_dir}/{run_name}.out.velz.*.pfb'))

        if not self.pressure_files:
            raise ValueError(f"No pressure files found for run {run_name} in {self.base_dir}")
        if not self.evaptrans_files:
            raise ValueError(f"No evaptrans files found for run {run_name} in {self.base_dir}")
        if not self.velx_files:
            raise ValueError(f"No velx files found for run {run_name} in {self.base_dir}")
        if not self.vely_files:
            raise ValueError(f"No vely files found for run {run_name} in {self.base_dir}")
        if not self.velz_files:
            raise ValueError(f"No velz files found for run {run_name} in {self.base_dir}")

        info(f"Found {len(self.pressure_files)} pressure files, {len(self.evaptrans_files)} evaptrans files, and {len(self.velx_files)} velocity files (velx/vely/velz)")

        # Determine maximum available timesteps accounting for sequence length
        max_available_timesteps = min(len(self.pressure_files), len(self.evaptrans_files), 
                                     len(self.velx_files), len(self.vely_files), len(self.velz_files)) - self.n_timesteps
        if max_available_timesteps <= 0:
            raise ValueError(f"Not enough timesteps for n_timesteps={self.n_timesteps}. Need at least {self.n_timesteps + 1} files.")

        # Pre-compute sizes from first file
        self.size_test = read_pfb(self.pressure_files[0])
        self.X_EXTENT = self.size_test.shape[2]
        self.Y_EXTENT = self.size_test.shape[1]
        self.Z_EXTENT = self.size_test.shape[0]
        self.T_EXTENT = max_available_timesteps

        verbose(f"Dataset dimensions: T={self.T_EXTENT}, Z={self.Z_EXTENT}, Y={self.Y_EXTENT}, X={self.X_EXTENT}")
        verbose(f"Using {self.n_timesteps}-timestep sequences")

        # Create static data dictionary to avoid loading the same static data multiple times
        self.static_data_dict = {}

        # Create batch generator for efficient indexing
        # For multi-timestep, we need n_timesteps+1 consecutive times (initial + n_timesteps targets)
        time_window_size = self.n_timesteps + 1

        self.dummy_data = xr.Dataset().assign_coords({
            'time': np.arange(self.T_EXTENT),
            'z': np.arange(self.Z_EXTENT),
            'y': np.arange(self.Y_EXTENT),
            'x': np.arange(self.X_EXTENT)
        })

        self.bgen = xb.BatchGenerator(
            self.dummy_data,
            input_dims={'x': self.patch_size_x, 'y': self.patch_size_y, 'time': time_window_size},
            input_overlap={'x': self.overlap_x, 'y': self.overlap_y, 'time': time_window_size-1},
            #return_partial=False,
            #shuffle=self.shuffle,
        )

        # Generate variable names
        self.generate_namelist()

        # Preload static parameters if requested
        if self.preload:
            self._preload_static_parameters()

    def generate_namelist(self):
        """
        Generate a list of names that will be used to input to the model.
        This will be used as a way to record the order that the variables
        go into the model so that they can be scaled internally.
        """
        self.PRESSURE_NAMES = [f'press_diff_{i}' for i in range(self.Z_EXTENT)]
        self.EVAPTRANS_NAMES = [f'evaptrans_{i}' for i in range(self.n_evaptrans)]
        self.VELOCITY_NAMES = [f'vel_diff_{i}' for i in range(3*self.Z_EXTENT)]
        self.PARAM_NAMES = []
        self.OUTPUT_NAMES = [f'press_diff_{i}' for i in range(self.Z_EXTENT)]

        # Use a tiny key just to look up what we need
        patch_keys = {'x': {'start': 0, 'stop': 2},
                      'y': {'start': 0, 'stop': 2},}

        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            file_name = f'{self.base_dir}/{self.run_name}.out.{parameter}.pfb'

            # param_temp shape is (n_layers, y, x)
            param_temp = read_pfb(file_name, keys=patch_keys)

            if param_temp.shape[0] == 1:
                self.PARAM_NAMES.append(parameter)
            else:
                temp_namelist = [f'{parameter}_{i}' for i in range(param_temp.shape[0])]
                # Select appropriate layers
                if n_lay > 0:
                    temp_namelist = temp_namelist[0:n_lay]
                elif n_lay < 0:
                    temp_namelist = temp_namelist[n_lay:]

                # Add the new names to the list
                self.PARAM_NAMES.extend(temp_namelist)

    def _preload_static_parameters(self):
        """Preload all static parameters into memory"""
        info("Preloading static parameters...")
        # Load static parameters sequentially
        for parameter in self.parameter_list:
            file_name = f'{self.base_dir}/{self.run_name}.out.{parameter}.pfb'
            self.static_data_dict[parameter] = read_pfb(file_name)
        verbose("Static parameters preloaded successfully.")

    def _read_cached_pfb(self, file_path, x_min=None, x_max=None, y_min=None, y_max=None):
        """Cache-aware PFB file reader that saves entire files and extracts patches from cache"""
        # Check if file is already in cache
        if file_path in self.cache:
            data = self.cache[file_path]
        else:
            # Read the entire file and store it in cache
            data = read_pfb(file_path)
            # Manage cache size
            if len(self.cache) >= self.cache_size:
                # Remove the least recently used item
                # (simple approach: remove first item)
                self.cache.pop(next(iter(self.cache)))
            # Store in cache
            self.cache[file_path] = data

        # Extract subset if needed
        if x_min is not None:
            data_slice = data[:, y_min:y_max+1, x_min:x_max+1]
            return data_slice
        return data

    def __len__(self):
        return len(self.bgen)

    def __getitem__(self, idx):
        sample_indices = self.bgen[idx]

        # Extract spatial and temporal indices from xbatcher
        time_indices = sample_indices['time'].values  # This is now a sequence
        x_min, x_max = sample_indices['x'].values[[0, -1]]
        y_min, y_max = sample_indices['y'].values[[0, -1]]

        # Create patch keys for read_pfb_sequence
        patch_keys = {
            'x': {'start': x_min, 'stop': x_max+1},
            'y': {'start': y_min, 'stop': y_max+1},
        }

        # Construct file sequences for pressure, evaptrans, and velocity
        pressure_file_sequence = [self.pressure_files[i] for i in time_indices]
        evaptrans_file_sequence = [self.evaptrans_files[i] for i in time_indices[1:]]  # Skip first for evaptrans (t+1, t+2, ...)
        velx_file_sequence = [self.velx_files[i] for i in time_indices] 
        vely_file_sequence = [self.vely_files[i] for i in time_indices] 
        velz_file_sequence = [self.velz_files[i] for i in time_indices]

        # Read pressure sequence (initial state + targets)
        pressure_sequence = read_pfb_sequence(pressure_file_sequence, keys=patch_keys)

        # Split into initial state and targets
        state_data = pressure_sequence[0]  # Initial state at time t
        target_sequence = pressure_sequence[1:]  # Target states at t+1, t+2, ...

        # Read evaptrans sequence (for t+1, t+2, ...)
        if evaptrans_file_sequence:
            evaptrans_sequence_raw = read_pfb_sequence(evaptrans_file_sequence, keys=patch_keys)

            # Process evaptrans layers for each timestep
            evaptrans_sequence = []
            for evaptrans in evaptrans_sequence_raw:
                if self.n_evaptrans > 0:
                    evaptrans = evaptrans[0:self.n_evaptrans,:,:]
                elif self.n_evaptrans < 0:
                    evaptrans = evaptrans[self.n_evaptrans:,:,:]
                evaptrans_sequence.append(evaptrans)
            evaptrans_sequence = np.array(evaptrans_sequence)
        else:
            # Single timestep case
            evaptrans_sequence = np.zeros((1, abs(self.n_evaptrans), y_max-y_min+1, x_max-x_min+1))

        # Read velocity sequence (for t+1, t+2, ...)
        # Create modified patch_keys for velocity files that have extra layers
        # velx needs one extra element in x direction
        velx_patch_keys = {
            'x': {'start': x_min, 'stop': x_max+2},  # +2 instead of +1 to include extra layer
            'y': {'start': y_min, 'stop': y_max+1},
        }
        # vely needs one extra element in y direction
        vely_patch_keys = {
            'x': {'start': x_min, 'stop': x_max+1},
            'y': {'start': y_min, 'stop': y_max+2},  # +2 instead of +1 to include extra layer
        }
        # velz uses same patch_keys (extra layer is in z, handled by diff)
        velx_sequence_raw = read_pfb_sequence(velx_file_sequence, keys=velx_patch_keys)
        vely_sequence_raw = read_pfb_sequence(vely_file_sequence, keys=vely_patch_keys)
        velz_sequence_raw = read_pfb_sequence(velz_file_sequence, keys=patch_keys)

        # Calculate differences across each dimension to normalize all velocities to same x,y dimensions
        # velx: (z, y, x + 1) with extra layer in x -> diff along x (axis=2) -> (z, y, x)
        # vely: (z, y + 1, x) with extra layer in y -> diff along y (axis=1) -> (z, y, x)
        # velz: (z + 1, y, x) with extra layer in z -> diff along z (axis=0) -> (z, y, x)
        velocity_sequence = []
        for t in range(len(velx_sequence_raw)):
            velx_t = velx_sequence_raw[t]  # Shape: (z, y, x + 1) e.g. (25, 25, 26)
            vely_t = vely_sequence_raw[t]  # Shape: (z, y + 1, x) e.g. (25, 26, 25)
            velz_t = velz_sequence_raw[t]  # Shape: (z + 1, y, x) e.g. (26, 25, 25)
            
            # Calculate differences along the dimension with extra layer
            velx_t = np.diff(velx_t, axis=2)  # Diff along x: (z, y, x + 1) -> (z, y, x) e.g. (25, 25, 25)
            vely_t = np.diff(vely_t, axis=1)  # Diff along y: (z, y + 1, x) -> (z, y, x) e.g. (25, 25, 25)
            velz_t = np.diff(velz_t, axis=0)  # Diff along z: (z + 1, y, x) -> (z, y, x) e.g. (25, 25, 25)
            
            # Concatenate along channel dimension: (z, y, x) -> (3*z, y, x)
            velocity_combined = np.concatenate([velx_t, vely_t, velz_t], axis=0)
            velocity_sequence.append(velocity_combined)
        velocity_sequence = np.array(velocity_sequence)

        # Load parameter data (static, same for all timesteps)
        parameter_data = []
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            file_name = f'{self.base_dir}/{self.run_name}.out.{parameter}.pfb'

            if self.preload:
                param_temp = self.static_data_dict[parameter]
                param_temp = param_temp[:, y_min:y_max+1, x_min:x_max+1]
            else:
                param_temp = read_pfb(file_name, keys=patch_keys)

            # Process layers
            if param_temp.shape[0] > 1:
                if n_lay > 0:
                    param_temp = param_temp[0:n_lay,:,:]
                elif n_lay < 0:
                    param_temp = param_temp[n_lay:,:,:]

            parameter_data.append(param_temp)

        parameter_data = np.concatenate(parameter_data, axis=0)

        # Convert to torch tensors
        state_data = torch.from_numpy(state_data).to(self.dtype).squeeze()
        parameter_data = torch.from_numpy(parameter_data).to(self.dtype).squeeze()
        target_sequence = torch.from_numpy(target_sequence).to(self.dtype).squeeze()
        evaptrans_sequence = torch.from_numpy(evaptrans_sequence).to(self.dtype).squeeze()
        velocity_sequence = torch.from_numpy(velocity_sequence).to(self.dtype).squeeze()

        # Clean invalid values
        state_data[state_data < -9999] = -10
        parameter_data[parameter_data < -9999] = -10
        target_sequence[target_sequence < -9999] = -10
        evaptrans_sequence[evaptrans_sequence < -9999] = -10
        velocity_sequence[velocity_sequence < -9999] = -10

        #TODO: FIXME
        self.flag_set = False
        if self.flag_set:
            info(f"State data shape: {state_data.shape}, min: {state_data.min()}, max: {state_data.max()}")
            info(f"Evaptrans sequence shape: {evaptrans_sequence.shape}")
            info(f"Velocity sequence shape: {velocity_sequence.shape}")
            info(f"Parameter data shape: {parameter_data.shape}")
            info(f"Target sequence shape: {target_sequence.shape}")

        return state_data, evaptrans_sequence, velocity_sequence, parameter_data, target_sequence

    def clear_cache(self):
        """Clear the internal file cache"""
        self.cache.clear()
