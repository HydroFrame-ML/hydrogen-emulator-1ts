## This folder contains notebooks for creating Subsets for training and testing the 1TS emulator. 

## Training on existing conus outputs: 
These scripts subset exsiting CONUS run outputs and create folders with static and transient outputs to use for model training. 
- `make_subset_domain_CONUS2.0.ipynb`
- `make_subset_domain_CONUS2.0.ipynb`

## Running watershed domains with ParFlow: 
To setup and run a watershed domain follow these steps: 

1. Find the outlet of your desired watershed using the HydroGEN app and then use `setup_watershed_run_CONUS2.1.ipynb` to define the domain and subset the inputs.
2. Modify the user input in `run_watershed.sh` and submit as a job `sbatch 
3. 
