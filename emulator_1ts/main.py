import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Support both ``python -m emulator_1ts.main`` and direct execution via
# ``python emulator_1ts/main.py``. In direct mode Python puts the package
# directory, rather than its parent, on sys.path and relative imports fail.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from emulator_1ts.dataset import ParFlowDataset
    from emulator_1ts.model import get_model
    from emulator_1ts.scalers import DEFAULT_SCALER_PATH
    from emulator_1ts.train import train_model
    from emulator_1ts.logger import (
        LogLevel,
        error,
        get_log_level,
        info,
        set_log_level,
        verbose,
    )
    from emulator_1ts.callbacks import CallbackManager, create_callbacks_from_config
    from emulator_1ts.experiment_tracking import create_tensorboard_tracker_from_config
    from emulator_1ts.utils import (
        calculate_metrics,
        get_dtype,
        get_loss,
        get_optimizer,
        get_scheduler,
    )
else:
    from .dataset import ParFlowDataset
    from .model import get_model
    from .scalers import DEFAULT_SCALER_PATH
    from .train import train_model
    from .logger import LogLevel, error, get_log_level, info, set_log_level, verbose
    from .callbacks import CallbackManager, create_callbacks_from_config
    from .experiment_tracking import create_tensorboard_tracker_from_config
    from .utils import calculate_metrics, get_dtype, get_loss, get_optimizer, get_scheduler


def read_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def custom_collate(batch):
    s, e, p, y, masks = [], [], [], [], []
    for b in batch:
        s.append(b[0])
        e.append(b[1])
        p.append(b[2])
        y.append(b[3])
        masks.append(b[4])

    s = torch.stack(s)
    p = torch.stack(p)

    # Handle both single-timestep and multi-timestep data
    if len(e[0].shape) == 4:  # Multi-timestep: [n_timesteps, z, y, x]
        # Stack along batch dimension, keeping timestep dimension first
        e = torch.stack(e, dim=1)  # [n_timesteps, batch, z, y, x]
        y = torch.stack(y, dim=1)  # [n_timesteps, batch, z, y, x]
        masks = torch.stack(masks, dim=1)
    else:  # Single-timestep: [z, y, x]
        e = torch.stack(e)
        y = torch.stack(y)
        masks = torch.stack(masks)

    return s, e, p, y, masks


def set_seed_all():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        # torch.use_deterministic_algorithms(True) - doesn't work for relection
        torch.backends.cudnn.benchmark = False


def dataset_definition_for_split(data_def, split, run_name, dtype):
    """Build dataset kwargs and route an optional member allowlist by split."""
    dataset_def = data_def.copy()
    location_key = f'{split}_data_location'
    member_key = f'{split}_member_ids'

    if location_key not in dataset_def:
        raise ValueError(f'Missing required data setting: {location_key}')

    dataset_def['data_location'] = dataset_def.pop(location_key)
    # By default the historical experiment ``name`` is also the PFB prefix,
    # but ensemble experiments may give the training run a distinct name.
    dataset_def['run_name'] = dataset_def.get('run_name', run_name)
    dataset_def['dtype'] = dtype

    if member_key in dataset_def:
        dataset_def['member_ids'] = dataset_def.pop(member_key)

    # Settings for other splits are configuration metadata, not dataset args.
    for other_split in ('train', 'validation', 'test'):
        if other_split != split:
            dataset_def.pop(f'{other_split}_data_location', None)
            dataset_def.pop(f'{other_split}_member_ids', None)

    return dataset_def


def validation_is_enabled(data_def):
    """Return whether the config requests a validation dataset.

    An explicitly empty ``validation_member_ids`` list is a convenient way to
    disable validation for quick overfitting/debug runs while retaining the
    validation data location in shared configs.
    """
    return (
        'validation_data_location' in data_def
        and data_def.get('validation_member_ids') != []
    )


def complete_model_definition(model_type, model_def, dataset):
    """Attach data-derived channel metadata and spatial dimensions."""

    expected_channels = (
        len(dataset.PRESSURE_NAMES)
        + len(dataset.EVAPTRANS_NAMES)
        + len(dataset.PARAM_NAMES)
    )
    configured_channels = model_def.get('in_channels')
    if configured_channels != expected_channels:
        raise ValueError(
            f"model_def.in_channels={configured_channels} does not match the "
            f"dataset's {expected_channels} channels "
            f"({len(dataset.PRESSURE_NAMES)} pressure + "
            f"{len(dataset.EVAPTRANS_NAMES)} evaptrans + "
            f"{len(dataset.PARAM_NAMES)} static)"
        )

    # Scalers are basin-specific. Without ``model_def.scalers`` every run
    # standardizes with the packaged CONUS2.1 CONUS-wide statistics, which are
    # the wrong distribution for a subset domain or a perturbed ensemble.
    if model_def.get('scalers') is None:
        info(
            "No model_def.scalers configured; using the packaged CONUS2.1 "
            f"statistics ({DEFAULT_SCALER_PATH})"
        )
    else:
        info(f"Using configured scalers: {model_def['scalers']}")

    model_def['pressure_names'] = dataset.PRESSURE_NAMES
    model_def['evaptrans_names'] = dataset.EVAPTRANS_NAMES
    model_def['param_names'] = dataset.PARAM_NAMES
    model_def['n_evaptrans'] = dataset.n_evaptrans
    model_def['parameter_list'] = dataset.parameter_list
    model_def['param_nlayer'] = dataset.param_nlayer
    if model_type == 'convnext_unet':
        # Resolve the schedule from the actual tensor/patch dimensions rather
        # than tying the architecture to one basin's dimensions in YAML.
        model_def['input_height'] = dataset.patch_size_y
        model_def['input_width'] = dataset.patch_size_x
    return model_def


def train(
    name: str,
    log_location: str,
    model_type: str,
    optimizer: str,
    loss,
    n_epochs: int,
    batch_size: int,
    lr: float,
    data_def: dict,
    model_def: dict,
    device: str,
    num_workers: int,
    dtype: str,
    set_seed: bool,
    config: dict,
    **kwargs
):
    info(f"Initializing training with name: {name}")
    verbose(f"Training parameters: epochs={n_epochs}, batch_size={batch_size}, lr={lr}, device={device}")

    # Use the configured output directory for all training artifacts and
    # TensorBoard logs. Create it before callbacks or final artifact saves run.
    Path(log_location).expanduser().mkdir(parents=True, exist_ok=True)

    if set_seed:
        set_seed_all()
        info(f"Setting random seed for reproducibility")

    # Create the data loaders
    dtype = get_dtype(dtype)
    info("Creating training dataset and data loader")
    train_data_def = dataset_definition_for_split(data_def, 'train', name, dtype)
    dataset = ParFlowDataset(**train_data_def)
    verbose(f"Training dataset created with {len(dataset)} samples")
    train_dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate,
        num_workers=num_workers//2,
        prefetch_factor=2
    )

    val_dl = None
    if validation_is_enabled(data_def):
        info("Creating validation dataset and data loader")
        validation_data_def = dataset_definition_for_split(
            data_def, 'validation', name, dtype
        )
        val_dataset = ParFlowDataset(**validation_data_def)
        verbose(f"Validation dataset created with {len(val_dataset)} samples")
        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate,
            num_workers=num_workers//2,
            prefetch_factor=2
        )
    elif data_def.get('validation_member_ids') == []:
        info("Validation disabled because validation_member_ids is empty")

    # Create the model
    info(f"Creating model of type: {model_type}")
    model_def = complete_model_definition(model_type, model_def, dataset)
    model = get_model(model_type, model_def)
    if model_type == 'convnext_unet':
        resolved_schedule = [
            [factor_y, factor_x]
            for factor_y, factor_x in zip(
                model.downsample_factor_y, model.downsample_factor_x
            )
        ]
        config['resolved_model'] = {
            'input_height': model.input_height,
            'input_width': model.input_width,
            'downsample_factors': resolved_schedule,
        }
        info(f"Resolved U-Net downsampling schedule: {resolved_schedule}")
    model = model.to(device).to(dtype)
    verbose(f"Model created and moved to {device} with dtype {dtype}")


    # Create the optimizer and loss function
    info(f"Setting up optimizer ({optimizer}) and loss function ({loss})")
    optimizer_obj = get_optimizer(optimizer, model, lr)
    # Pass the model so a residual-aware loss can read its per-layer pressure
    # sigmas; the loss carries buffers, so it moves to the training device.
    loss_fn = get_loss(loss, model=model).to(device)

    # Create learning rate scheduler if specified
    scheduler = None
    if 'callbacks' in config and 'lr_scheduler' in config['callbacks']:
        lr_config = config['callbacks']['lr_scheduler']
        if lr_config.get('enabled', False):
            scheduler_type = lr_config.get('type', 'ReduceLROnPlateau')
            scheduler = get_scheduler(scheduler_type, optimizer_obj, **lr_config)
            info(f"Learning rate scheduler created: {scheduler_type}")

    # Extract multi-timestep training parameters
    autoregressive_loss_weights = None
    if 'autoregressive' in config and config['autoregressive'] is not None:
        autoregressive_config = config['autoregressive']
        autoregressive_loss_weights = autoregressive_config.get('loss_weights', None)
        if autoregressive_loss_weights:
            info(f"Using custom autoregressive loss weights: {autoregressive_loss_weights}")

    # Create callback manager
    callback_manager = CallbackManager()

    # Add callbacks from config
    callbacks = create_callbacks_from_config(config, model, log_location, name)
    for callback in callbacks:
        callback_manager.add_callback(callback)

    # Add TensorBoard tracker
    tensorboard_tracker = create_tensorboard_tracker_from_config(
        config, name, log_dir=log_location
    )
    if tensorboard_tracker:
        callback_manager.add_callback(tensorboard_tracker)
        info("TensorBoard tracking enabled")

    info("Starting model training")
    metrics = train_model(
        model,
        train_dl,
        optimizer_obj,
        loss_fn,
        n_epochs,
        scheduler=scheduler,
        val_dl=val_dl,
        callback_manager=callback_manager,
        device=device,
        dtype=dtype,
        autoregressive_loss_weights=autoregressive_loss_weights
    )
    info("Training completed, displaying metrics")
    print('----------------------------------------')
    print(metrics)
    print('----------------------------------------')

    info("Saving model artifacts")
    metrics_filename = f'{log_location}/{name}_metrics.csv'
    weights_filename = f'{log_location}/{name}_weights_only.pth'
    model_filename = f'{log_location}/{name}_model.pth'
    config['model_path'] = model_filename
    config['weights_path'] = weights_filename
    config['metrics_path'] = metrics_filename

    verbose(f"Saving config to {log_location}/{name}_config.yaml")

    # NOTE: remove timesteps because we only want to use the model 1ts
    config["data_def"].pop("n_timesteps")
    with open(f'{log_location}/{name}_config.yaml', 'w') as f:
        yaml.safe_dump(config, f)

    verbose(f"Saving metrics to {metrics_filename}")
    metrics.to_csv(metrics_filename)

    #model = model.to(device='cpu')

    verbose(f"Saving model weights to {weights_filename}")
    torch.save(model.state_dict(), weights_filename)

    verbose(f"Creating and saving TorchScript model to {model_filename}")

    m = torch.jit.script(model)
    torch.jit.save(m, model_filename)

    info("Training process completed successfully")
    print('----------------------------------------')
    print(f'Metrics saved to {metrics_filename}')
    print(f'Model saved to {model_filename}')
    print(f'Config saved to {log_location}/{name}_config.yaml')


def export_model(config, weights_path=None, output_path=None):
    """Build and script a trained model in a fresh Python process.

    Training writes both the completed model definition and the state dict
    before its final TorchScript export. This entry point can therefore recover
    an export if scripting failed after a long training run, without retraining.
    """

    configured_weights = weights_path or config.get('weights_path')
    configured_output = output_path or config.get('model_path')
    if not configured_weights:
        raise ValueError("No weights path supplied and config has no weights_path")
    if not configured_output:
        raise ValueError("No output path supplied and config has no model_path")

    weights_path = Path(configured_weights).expanduser().resolve()
    output_path = Path(configured_output).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {weights_path}")

    model = get_model(config['model_type'], dict(config['model_def']))
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device='cpu', dtype=get_dtype(config.get('dtype', 'float32')))
    model.eval()
    scripted = torch.jit.script(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, output_path)
    info(f"TorchScript model exported to {output_path}")
    return output_path



def test(
    name: str,
    log_location: str,
    model_path: str,
    data_def: dict,
    batch_size: int,
    device: str,
    num_workers: int,
    dtype: str,
    save_inputs: bool,
    **kwargs
):
    info(f"Initializing testing with name: {name}")
    verbose(f"Testing parameters: batch_size={batch_size}, device={device}")

    dtype = get_dtype(dtype)
    # Load the model
    info(f"Loading model from {model_path}")
    model = torch.jit.load(model_path)
    model = model.to(device).to(dtype)
    verbose(f"Model loaded and moved to {device} with dtype {dtype}")

    # Create the data loader
    info("Creating dataset and data loader for testing")
    test_data_def = dataset_definition_for_split(data_def, 'test', name, dtype)
    dataset = ParFlowDataset(**test_data_def)
    verbose(f"Test dataset created with {len(dataset)} samples")
    test_dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=False,
        num_workers=num_workers,
    )

    info("Starting model evaluation")
    model.eval()
    all_outputs = []
    all_targets = []
    all_valid_masks = []

    if save_inputs:
        all_states = []
        all_evaptrans = []
        all_scaled_states = []
        all_scaled_evaptrans = []


    verbose("Processing test batches")

    # Use tqdm progress bar only in verbose mode
    is_verbose = get_log_level() == LogLevel.VERBOSE

    # Wrap test_dl with tqdm if in verbose mode
    batch_iterator = tqdm(test_dl, desc="Testing batch") if is_verbose else test_dl

    with torch.no_grad():
        for i, batch in enumerate(batch_iterator):
            s, e, p, y, valid_mask = batch

            if save_inputs:
                all_states.append(s)
                all_evaptrans.append(e)
                if i == 0:
                    all_parameters = p

            s = s.to(device=device, non_blocking=True)
            e = e.to(device=device, non_blocking=True)
            p = p.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            valid_mask = valid_mask.to(device=device, non_blocking=True)

            model.scale_pressure(s)
            model.scale_evaptrans(e)
            model.scale_statics(p)
            model.scale_pressure(y)
            s.masked_fill_(~valid_mask, 0)
            y.masked_fill_(~valid_mask, 0)

            if save_inputs:
                all_scaled_states.append(s.cpu())
                all_scaled_evaptrans.append(e.cpu())
                if i == 0:
                    all_scaled_parameters = p.cpu()

            outputs = model(s, e, p)

            # Unscale the outputs
            model.unscale_pressure(outputs)
            model.unscale_pressure(y)

            all_outputs.append(outputs.cpu())
            all_targets.append(y.cpu())
            all_valid_masks.append(valid_mask.cpu())

    info("Evaluation completed, processing results")
    if save_inputs:
        all_states = torch.cat(all_states)
        all_evaptrans = torch.cat(all_evaptrans)
        all_scaled_states = torch.cat(all_scaled_states)
        all_scaled_evaptrans = torch.cat(all_scaled_evaptrans)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    all_valid_masks = torch.cat(all_valid_masks)
    info(f'All outputs shape: {all_outputs.shape}')
    info(f'All targets shape: {all_targets.shape}')

    # Save the outputs
    output_filename = f'{log_location}/{name}_outputs.pt'
    verbose(f"Saving model outputs to {output_filename}")
    torch.save(all_outputs, output_filename)
    info(f'Outputs saved to {output_filename}')
    # Save the targets
    target_filename = f'{log_location}/{name}_targets.pt'
    verbose(f"Saving targets to {target_filename}")
    torch.save(all_targets, target_filename)
    info(f'Targets saved to {target_filename}')
    mask_filename = f'{log_location}/{name}_valid_masks.pt'
    verbose(f"Saving valid-cell masks to {mask_filename}")
    torch.save(all_valid_masks, mask_filename)
    info(f'Valid-cell masks saved to {mask_filename}')
    if save_inputs:
        # Save the states
        states_filename = f'{log_location}/{name}_states.pt'
        verbose(f"Saving states to {states_filename}")
        torch.save(all_states, states_filename)
        info(f'States saved to {states_filename}')
        # Save the evapotranspiration
        evaptrans_filename = f'{log_location}/{name}_evaptrans.pt'
        verbose(f"Saving evapotranspiration to {evaptrans_filename}")
        torch.save(all_evaptrans, evaptrans_filename)
        info(f'Evapotranspiration saved to {evaptrans_filename}')
        # Save the parameters
        parameters_filename = f'{log_location}/{name}_parameters.pt'
        verbose(f"Saving parameters to {parameters_filename}")
        torch.save(all_parameters, parameters_filename)
        info(f'Parameters saved to {parameters_filename}')
        # Save the scaled states
        scaled_states_filename = f'{log_location}/{name}_scaled_states.pt'
        verbose(f"Saving scaled states to {scaled_states_filename}")
        torch.save(all_scaled_states, scaled_states_filename)
        info(f'Scaled states saved to {scaled_states_filename}')
        # Save the scaled evapotranspiration
        scaled_evaptrans_filename = f'{log_location}/{name}_scaled_evaptrans.pt'
        verbose(f"Saving scaled evapotranspiration to {scaled_evaptrans_filename}")
        torch.save(all_scaled_evaptrans, scaled_evaptrans_filename)
        info(f'Scaled evapotranspiration saved to {scaled_evaptrans_filename}')
        # Save the scaled parameters
        scaled_parameters_filename = f'{log_location}/{name}_scaled_parameters.pt'
        verbose(f"Saving scaled parameters to {scaled_parameters_filename}")
        torch.save(all_scaled_parameters, scaled_parameters_filename)
        info(f'Scaled parameters saved to {scaled_parameters_filename}')

    # Calculate and print metrics
    info("Calculating evaluation metrics")
    metrics = calculate_metrics(all_outputs, all_targets, mask=all_valid_masks)
    metrics_filename = f'{log_location}/{name}_test_metrics.csv'
    verbose(f"Saving metrics to {metrics_filename}")
    metrics.to_csv(metrics_filename)
    info(f'Test metrics saved to {metrics_filename}')
    info("Test results:")
    print(metrics)

    info("Testing process completed successfully")


def main(config, mode, log_level, save_inputs, weights_path=None, output_path=None):
    # Set the log level
    set_log_level(log_level)

    # Log the start of the program
    info(f"Starting emulator in {mode} mode with log level {log_level}")

    # Read the configuration file
    config = read_config(config)
    verbose(f"Loaded configuration from {config}")

    if mode == "train":
        info("Starting training process")
        train(**config, config=config)
    elif mode == "test":
        info("Starting testing process")
        test(**config, save_inputs=save_inputs)
    elif mode == "export":
        info("Starting fresh-process TorchScript export")
        export_model(config, weights_path=weights_path, output_path=output_path)


if __name__ == "__main__":
    # EXAMPLE USAGE: python main.py --config example_config.yaml --mode train --log-level info
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test", "export"],
        default="train",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="State-dict path for export mode; defaults to config.weights_path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="TorchScript path for export mode; defaults to config.model_path",
    )
    parser.add_argument(
        "--log-level", type=str, choices=["silent", "info", "verbose"], default="silent",
        help="Set the logging level (silent, info, verbose)"
    )
    parser.add_argument(
        "--save_inputs",
        action="store_true",
        help="If set, saves inputs during testing."
    )
    args = parser.parse_args()
    save_inputs = args.save_inputs if args.mode == "test" else False
    main(
        args.config,
        args.mode,
        args.log_level,
        save_inputs,
        weights_path=args.weights,
        output_path=args.output,
    )
