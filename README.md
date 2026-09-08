# 1-Timestep emulator

This folder contains the scripts for the one time step emulator. The goal of this emulator is to predict the next timestep for a ParFlow simulation. 

## Key files in the `emulator-1ts` folder: 
- `main.py`: this is the main script that does the training. It call all the other classes and functions.
- `*_scalers*.yaml`: Are the scalers files with the mean and standard deviation for every layer of every variable. Right now everything is set to `standard_scaler` meaning that variables will be scaled by subtracting the mean and dividing by the standard deviation. All scalers were calculated using scripts in the `CONUS2_Data_Prep` folder, refer to the readme there for more details. There are multiple versions of the scalers files available using the following naming convention: 
    - `*original` or `adjusted` : The original calcualted scalers with no adjustments made. The adjusted files have the standard deviation changed to 1 for all of the layers where the standard devaition was 0 or something <1e-15.
    - `*_pressure.yaml`: The _pressure versions have pressure scalers calcualted based on the  pressure files themselves instead of pressure file differences between timesteps. *NOTE* These are still called 'press_diff' scalers in the yaml file to match whats expected in line 90 of 'dataset.py` where the scaling happens. Should make this an option later.
    - `CONUS21_*` or `default_*`: The CONUS21 versions have the evaptrans and pressure scalers calculated from the CONUS2.1 run WY2003V2. The `default` versions use the CONUS2 Baseline run for the evaptrans and pressure scalers. 

## Before you start: 
- In order to run a training run you first need to generate a set of test data. You can do that using `notebooks/make_subset_domain*.ipynb`
- `example_config.yaml`: Contains all of the settings needed for a run. These are the knobs that should be turned.
- You will also need to adjust the `example.config.yaml` to reflect your local paths and run names.
    - **Note**: The`in_channels` should equal the total number of layers you are using from your parameter list + n_evaptrans (# of evaptrans layers being used) + 10 (#of layers in a perssure file). (For example, if your input parameter list is slopex, slopey and permeabilityx and you use all layers from these files and have 4 evaptrans layers  then the in_channels will be 1+1+10+4+10 = 26)


## Setup on Verde
To run on verde: 
1. Lauch an interactive *Jupyter Lab* session and selece `hydrogen-shared` as the anaconda version.

*To run python scripts:* 
1. Start a terminal session from jupyter lab (`file/new/termnial`)
2. If the terminal prompt says `(base) (hydrogen-shared)` you will need to deactivate the base enviroment with `conda deactivate` you should then just see the prompt say `(hydrogen-shared)`

*To run Jupyter Notebooks:* 
Select `Python3` as the kernel and then you shoudl be good to run

*Note:* From other enviroments you can also use `module load hydrogen-shared` to load this enviroment. 

## How to run a traning run
From terminal: `python -m emulator_1ts.main --config example_config.yaml --mode train`

Direct execution is also supported: `python ./emulator_1ts/main.py --config example_config.yaml --mode train`

### Recover a TorchScript export without retraining

Training saves the completed config and `*_weights_only.pth` state dict before
creating TorchScript. If that final scripting step fails (for example, because
`model.py` changed while a long-running Python process was training), rerun only
the export in a fresh process:

```bash
uv run python -m emulator_1ts.main \
  --config runs/<experiment>_config.yaml \
  --mode export
```

The command uses `weights_path` and `model_path` from the completed config.
They can be overridden with `--weights` and `--output`.

### Fit scalers for your own basin

The packaged scalers describe CONUS2.1 over WY2003. Training a subset basin or a
perturbed ensemble against them standardizes with the wrong distribution, so fit
a matching set first. The easiest route reads the training config, so the fit
covers exactly the run, statics, and training members that will be trained on:

```bash
uv run python -m emulator_1ts.fit_scalers \
  --config convnext_unet_multistep_config_mjb_ensemble.yaml \
  --output mjb_scalers.yaml
```

Or point it at a directory directly. A single run directory and an ensemble root
of `member_*` directories both work:

```bash
uv run python -m emulator_1ts.fit_scalers \
  --data-location /path/to/ensemble --run-name mjb \
  --parameters perm_x perm_y porosity mannings mask \
  --member-ids 0000 0001 0002 \
  --timestep-interval 5 --output mjb_scalers.yaml
```

Then point the model at the result:

```yaml
model_def:
  scalers: mjb_scalers.yaml
```

Notes:

- **Fit on training members only.** Including validation or test members leaks
  their distribution into the model's inputs.
- Statistics cover active cells only, using the domain mask when one is present
  and dropping ParFlow sentinels either way.
- `--timestep-interval` strides over timesteps. A prime stride avoids sampling
  the same hour of day repeatedly, which is what the packaged CONUS statistics
  did; `--max-timesteps` caps the work for a quick look.
- A channel that is constant over the domain (`ssat` and `vg_n` are, on CONUS)
  gets `std: 0`, which the loader turns into `std: 1` with a warning. That is
  expected, and it is the same substitution the packaged `*_adjusted*` files
  make by hand.

## Training from a ParFlow ensemble

Set each split's data location to the ensemble root (the directory containing
`metadata.csv` and `member_*`) and select complete members for each split. The
PFB run prefix can be specified independently from the experiment name:

```yaml
name: mjb-resnet-initial

data_def:
  train_data_location: /path/to/data/mjb/ensemble
  validation_data_location: /path/to/data/mjb/ensemble
  test_data_location: /path/to/data/mjb/ensemble
  run_name: mjb

  train_member_ids: ['0000', '0001', '0002', '0003', '0004', '0005', '0006']
  validation_member_ids: ['0007']
  test_member_ids: ['0008', '0009']

  parameters:
    - [perm_x, 0]
  patch_size_x: 107
  patch_size_y: 89
  overlap_x: 0
  overlap_y: 0
  n_evaptrans: -4
  n_timesteps: 12
```

For a horizon of `H`, a sample starting at timestep `t` uses pressure at `t`,
evapotranspiration at `t+1 ... t+H`, and pressure targets at `t+1 ... t+H`.
Windows are constructed independently inside each member and incomplete final
windows are excluded; they never wrap into the next member.

Pressure cells containing ParFlow's no-data sentinel are tracked with a boolean
valid-cell mask. Inactive cells are excluded from training, validation, and test
metrics, and are reset to a neutral scaled value during autoregressive rollout.

## Multiscale ConvNeXT U-Net

`convnext_unet` adds a mask-aware encoder/decoder while retaining the current
ConvNeXT residual pressure prediction. A complete MJB example is provided in
`convnext_unet_multistep_config_mjb_ensemble.yaml`.

With `downsample_mode: auto`, the model derives an anisotropic pooling schedule
from the dataset patch dimensions. An axis is halved only if it remains at least
`min_coarse_cells` wide. For example, four levels with a minimum of eight cells
resolve `32 x 256` to `[(2,2), (2,2), (1,2), (1,2)]`. Decoder features are
resized to the exact skip shape, so odd dimensions and non-square basins do not
require padding to powers of two. The resolved schedule is logged and saved in
the completed experiment configuration.

The static `mask` parameter is required. Downsampling uses mask-normalized
pooling, and hidden/output features are reset outside the active domain at every
scale.

## CONUS2.1 Update progress: 
- New scalers have been calculated for the CONUS2.1 run and are available in this folder.
- The notebooks folder has a subset domain routine which is ready to use`make_subset_domain_CONUS21.ipynb` but before it will work WY2003V2 needs to be added to the data catalog and then the `transient_dataset` name will need to be changed to point to this. 

## Other things to add/change: 
1. We need to setup a testing run. A good first test would be the same locaton but a different point in time (we have little expectation that it will do good on a different location just yet since we are training on a very small subset)
1. Change the inputs so the number of layers used and the parameter list is a dictionary and not two separate lists.
2. Make a copy of the config file where the model is saved for documentation purposes.
