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
        cache_size=64,
        max_timesteps=None,  # Maximum timesteps for progressive training
        noise_std=1e-9,  # Standard deviation for Gaussian noise perturbation
        **kwargs,
    ):
        super().__init__()
        self.base_dir = data_location
        self.run_name = run_name
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.n_evaptrans = n_evaptrans
        self.n_timesteps = n_timesteps
        self.max_timesteps = max_timesteps or n_timesteps  # Use max if provided
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.shuffle = shuffle
        self.dtype = dtype
        self.preload = preload
        self.noise_std = noise_std

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

        if not self.pressure_files:
            raise ValueError(f"No pressure files found for run {run_name} in {self.base_dir}")
        if not self.evaptrans_files:
            raise ValueError(f"No evaptrans files found for run {run_name} in {self.base_dir}")

        info(f"Found {len(self.pressure_files)} pressure files and {len(self.evaptrans_files)} evaptrans files")

        # Determine maximum available timesteps accounting for sequence length
        # Use max_timesteps for validation to ensure we have enough data for progressive training
        max_needed_timesteps = max(self.max_timesteps, self.n_timesteps)
        max_available_timesteps = min(len(self.pressure_files), len(self.evaptrans_files)) - max_needed_timesteps
        if max_available_timesteps <= 0:
            raise ValueError(f"Not enough timesteps for max_timesteps={max_needed_timesteps}. Need at least {max_needed_timesteps + 1} files.")

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

        # Construct file sequences for pressure and evaptrans
        pressure_file_sequence = [self.pressure_files[i] for i in time_indices]
        evaptrans_file_sequence = [self.evaptrans_files[i] for i in time_indices[1:]]  # Skip first for evaptrans (t+1, t+2, ...)

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

        # Clean invalid values
        state_data[state_data < -9999] = -10
        parameter_data[parameter_data < -9999] = -10
        target_sequence[target_sequence < -9999] = -10
        evaptrans_sequence[evaptrans_sequence < -9999] = -10

        # Add small Gaussian noise perturbation to prevent numerical issues
        if self.noise_std > 0:
            # Add noise to state data
            noise = torch.randn_like(state_data) * self.noise_std
            state_data = state_data + noise

            # Add noise to parameter data
            noise = torch.randn_like(parameter_data) * self.noise_std
            parameter_data = parameter_data + noise

            # Add noise to target sequence
            noise = torch.randn_like(target_sequence) * self.noise_std
            target_sequence = target_sequence + noise

            # Add noise to evaptrans sequence
            noise = torch.randn_like(evaptrans_sequence) * self.noise_std
            evaptrans_sequence = evaptrans_sequence + noise

        #TODO: FIXME
        self.flag_set = False
        if self.flag_set:
            info(f"State data shape: {state_data.shape}, min: {state_data.min()}, max: {state_data.max()}")
            info(f"Evaptrans sequence shape: {evaptrans_sequence.shape}")
            info(f"Parameter data shape: {parameter_data.shape}")
            info(f"Target sequence shape: {target_sequence.shape}")

        return state_data, evaptrans_sequence, parameter_data, target_sequence

    def update_timesteps(self, new_timesteps: int):
        """
        Update the number of timesteps for progressive training.
        This recreates the batch generator with the new timestep settings.
        
        Args:
            new_timesteps: New number of timesteps to use
        """
        if new_timesteps == self.n_timesteps:
            verbose(f"Timesteps already set to {new_timesteps}, no update needed")
            return
        
        if new_timesteps > self.max_timesteps:
            raise ValueError(f"Cannot set timesteps to {new_timesteps}, exceeds max_timesteps={self.max_timesteps}")
        
        old_timesteps = self.n_timesteps
        self.n_timesteps = new_timesteps
        
        verbose(f"Updating dataset timesteps: {old_timesteps} -> {new_timesteps}")
        
        # Recreate batch generator with new time window size
        time_window_size = self.n_timesteps + 1
        
        self.bgen = xb.BatchGenerator(
            self.dummy_data,
            input_dims={'x': self.patch_size_x, 'y': self.patch_size_y, 'time': time_window_size},
            input_overlap={'x': self.overlap_x, 'y': self.overlap_y, 'time': time_window_size-1},
        )
        
        verbose(f"Dataset batch generator updated for {new_timesteps}-timestep sequences")
    
    def get_timesteps(self) -> int:
        """Get the current number of timesteps."""
        return self.n_timesteps
    
    def get_max_timesteps(self) -> int:
        """Get the maximum number of timesteps supported."""
        return self.max_timesteps

    def clear_cache(self):
        """Clear the internal file cache"""
        self.cache.clear()
