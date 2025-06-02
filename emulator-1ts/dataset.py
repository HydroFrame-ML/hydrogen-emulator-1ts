import xarray as xr
import os
import torch
import xbatcher as xb
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from parflow.tools.io import read_pfb
from torch.utils.data import Dataset
from functools import lru_cache
import concurrent.futures

from logger import info, verbose, error

class ParFlowDataset(Dataset):

    def __init__(
        self, data_location, run_name,
        parameter_list, 
        patch_size_x, 
        patch_size_y,
        overlap_x,
        overlap_y,
        param_nlayer, n_evaptrans=0,
        shuffle=False, dtype=torch.float32,
        preload=True, cache_size=64, **kwargs,
    ):
        super().__init__()
        self.base_dir = data_location
        self.run_name = run_name
        self.parameter_list = parameter_list
        self.param_nlayer = param_nlayer
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.n_evaptrans = n_evaptrans
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.shuffle = shuffle
        self.dtype = dtype
        self.preload = preload
        # NOTE: Used for internal debugging, should be set to False to disable
        self.flag_set = False
        
        # Cache for frequently accessed PFB files
        self.cache = {}
        self.cache_size = cache_size
        
        # Find and organize pressure files - note: getting rid of 00000 file 
        # because the evaptrans file is not included as initial condition
        self.pressure_files = sorted(glob(f'{self.base_dir}/{run_name}.out.press.*.pfb'))[1:]
        info(self.pressure_files[0])
        self.pressure_files = {
            't': self.pressure_files[0:-1],
            't+1': self.pressure_files[1:]
        }
        
        # Pre-compute sizes
        self.size_test = read_pfb(self.pressure_files['t'][0])
        self.X_EXTENT = self.size_test.shape[2] 
        self.Y_EXTENT = self.size_test.shape[1]
        self.Z_EXTENT = self.size_test.shape[0]
        self.T_EXTENT = len(self.pressure_files['t'])
        
        # Pre-compute evaptrans file paths that correspond to pressure files
        self.evaptrans_files = [f.replace('press', 'evaptrans') for f in self.pressure_files['t']]
        info(self.evaptrans_files[0])
        
        # Create static data dictionary to avoid loading the same static data multiple times
        self.static_data_dict = {}
        
        # Create batch generator for efficient indexing
        self.dummy_data = xr.Dataset().assign_coords({
            'time': np.arange(self.T_EXTENT),
            'z': np.arange(self.Z_EXTENT),
            'y': np.arange(self.Y_EXTENT),
            'x': np.arange(self.X_EXTENT)
        })
        
        self.bgen = xb.BatchGenerator(
            self.dummy_data,
            input_dims={'x': self.patch_size_x, 'y': self.patch_size_y, 'time': 1},
            input_overlap={'x': self.overlap_x, 'y': self.overlap_y},
            return_partial=False,
            shuffle=self.shuffle,
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

        # Extract indices
        time_index = sample_indices['time'].values[0]
        x_min, x_max = sample_indices['x'].values[[0, -1]]
        y_min, y_max = sample_indices['y'].values[[0, -1]]

        # Create patch keys
        patch_keys = {
            'x': {'start': x_min, 'stop': x_max+1},
            'y': {'start': y_min, 'stop': y_max+1},
        }
    
        # Load state data
        file_to_read = self.pressure_files['t'][time_index]
        state_data = self._read_cached_pfb(
            file_to_read, 
            x_min=x_min, x_max=x_max, 
            y_min=y_min, y_max=y_max
        )

        # Load target data
        file_to_read_target = self.pressure_files['t+1'][time_index]
        target_data = self._read_cached_pfb(
            file_to_read_target, 
            x_min=x_min, x_max=x_max, 
            y_min=y_min, y_max=y_max
        )

        # Load parameter data
        parameter_data = []
        for (parameter, n_lay) in zip(self.parameter_list, self.param_nlayer):
            # Get file path
            file_name = f'{self.base_dir}/{self.run_name}.out.{parameter}.pfb'
            
            # Load parameter data (either from preloaded cache or from disk)
            if self.preload:
                param_temp = self.static_data_dict[parameter]
                # Extract subset
                param_temp = param_temp[:, y_min:y_max+1, x_min:x_max+1]
            else:
                param_temp = self._read_cached_pfb(
                    file_name, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
                )

            # Process layers
            if param_temp.shape[0] > 1:
                if n_lay > 0:
                    param_temp = param_temp[0:n_lay,:,:]
                elif n_lay < 0:
                    param_temp = param_temp[n_lay:,:,:]

            parameter_data.append(param_temp)

        # Concatenate parameters
        parameter_data = np.concatenate(parameter_data, axis=0)

        # Load evaptrans data
        file_name_et = self.evaptrans_files[time_index]
        evaptrans = self._read_cached_pfb(
            file_name_et, 
            x_min=x_min, x_max=x_max, 
            y_min=y_min, y_max=y_max
        )
        
        # Process evaptrans layers
        if self.n_evaptrans > 0:
            evaptrans = evaptrans[0:self.n_evaptrans,:,:]
        elif self.n_evaptrans < 0:
            evaptrans = evaptrans[self.n_evaptrans:,:,:]
        
        # Convert to torch tensors efficiently
        # Use non_blocking=True if data is on a CUDA device
        state_data = torch.from_numpy(state_data).to(self.dtype)
        evaptrans = torch.from_numpy(evaptrans).to(self.dtype)
        parameter_data = torch.from_numpy(parameter_data).to(self.dtype)
        target_data = torch.from_numpy(target_data).to(self.dtype)

        # Convert nan to -10
        state_data = torch.nan_to_num(state_data, nan=-10.0)
        evaptrans = torch.nan_to_num(evaptrans, nan=-10.0)
        parameter_data = torch.nan_to_num(parameter_data, nan=-10.0)
        target_data = torch.nan_to_num(target_data, nan=-10.0)

        if self.flag_set:
            # Print statistics for debugging
            info(f"State data shape: {state_data.shape}, min: {state_data.min()}, max: {state_data.max()}")
            info(f"Evaptrans shape: {evaptrans.shape}, min: {evaptrans.min()}, max: {evaptrans.max()}")
            info(f"Parameter data shape: {parameter_data.shape}, min: {parameter_data.min()}, max: {parameter_data.max()}")
            info(f"Target data shape: {target_data.shape}, min: {target_data.min()}, max: {target_data.max()}")

        
        return state_data, evaptrans, parameter_data, target_data
        
    def clear_cache(self):
        """Clear the internal file cache"""
        self.cache.clear()
