import os
import yaml
import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from .dataset import ParFlowDataset
from .model import get_model
from .train import train_model
from .logger import set_log_level, info, verbose, error, LogLevel, get_log_level
from .callbacks import CallbackManager, create_callbacks_from_config
from .experiment_tracking import create_tensorboard_tracker_from_config
from .utils import get_optimizer, get_loss, get_dtype, calculate_metrics, get_scheduler

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def custom_collate(batch):
    s, e, p, y = [], [], [], []
    for b in batch:
        s.append(b[0])
        e.append(b[1])
        p.append(b[2])
        y.append(b[3])

    s = torch.stack(s)
    p = torch.stack(p)

    # Handle both single-timestep and multi-timestep data
    if len(e[0].shape) == 4:  # Multi-timestep: [n_timesteps, z, y, x]
        # Stack along batch dimension, keeping timestep dimension first
        e = torch.stack(e, dim=1)  # [n_timesteps, batch, z, y, x]
        y = torch.stack(y, dim=1)  # [n_timesteps, batch, z, y, x]
    else:  # Single-timestep: [z, y, x]
        e = torch.stack(e)
        y = torch.stack(y)

    return s, e, p, y


def set_seed_all(rank: int = 0):
    torch.manual_seed(0 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0 + rank)
        torch.cuda.manual_seed_all(0 + rank)
        # torch.use_deterministic_algorithms(True) - doesn't work for relection
        torch.backends.cudnn.benchmark = False


def _init_distributed():
    """
    Initialize torch.distributed when launched with torchrun (WORLD_SIZE > 1).
    Returns (distributed, rank, world_size, local_rank).
    """
    world_size_env = os.environ.get("WORLD_SIZE")
    if world_size_env is None or int(world_size_env) <= 1:
        return False, 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requested but CUDA is not available.")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, world_size, local_rank


def train(
    name: str,
    log_location: str,
    model_type: str,
    optimizer: str,
    loss: str,
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
    distributed, rank, world_size, local_rank = _init_distributed()
    is_main = rank == 0
    dist_cfg = config.get("distributed") or {}
    scale_lr_linear = bool(dist_cfg.get("scale_lr_linear", False))

    if distributed:
        train_device = torch.device("cuda", local_rank)
        eff_lr = lr * world_size if scale_lr_linear else lr
        if is_main:
            info(
                f"Distributed training: world_size={world_size}, per-GPU batch_size={batch_size}, "
                f"effective global batch_size={batch_size * world_size}"
            )
            if scale_lr_linear:
                info(f"Learning rate scaled linearly with world_size: base_lr={lr} -> eff_lr={eff_lr}")
    else:
        train_device = torch.device(device)
        eff_lr = lr

    if is_main:
        info(f"Initializing training with name: {name}")
        verbose(
            f"Training parameters: epochs={n_epochs}, batch_size={batch_size}, lr={eff_lr}, device={train_device}"
        )

    if set_seed:
        set_seed_all(rank)
        if is_main:
            info("Setting random seed for reproducibility")

    dtype = get_dtype(dtype)
    if is_main:
        info("Creating training dataset and data loader")
    train_data_def = data_def.copy()
    train_data_location = train_data_def.pop('train_data_location')
    train_data_def['data_location'] = train_data_location
    train_data_def['run_name'] = name
    dataset = ParFlowDataset(**train_data_def, dtype=dtype)
    if is_main:
        verbose(f"Training dataset created with {len(dataset)} samples")

    train_sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    nw_train = max(num_workers // 2, 0)
    train_dl_kw = dict(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=custom_collate,
        num_workers=nw_train,
        sampler=train_sampler,
        shuffle=train_sampler is None,
    )
    if nw_train > 0:
        train_dl_kw["prefetch_factor"] = 2
    train_dl = DataLoader(**train_dl_kw)

    val_dl = None
    val_sampler = None
    if 'validation_data_location' in data_def:
        if is_main:
            info("Creating validation dataset and data loader")
        validation_data_def = data_def.copy()
        validation_data_location = validation_data_def.pop('validation_data_location')
        validation_data_def['data_location'] = validation_data_location
        validation_data_def['run_name'] = name
        val_dataset = ParFlowDataset(**validation_data_def, dtype=dtype)
        if is_main:
            verbose(f"Validation dataset created with {len(val_dataset)} samples")
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
        nw_val = max(num_workers // 2, 0)
        val_dl_kw = dict(
            dataset=val_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate,
            num_workers=nw_val,
            sampler=val_sampler,
            shuffle=False,
        )
        if nw_val > 0:
            val_dl_kw["prefetch_factor"] = 2
        val_dl = DataLoader(**val_dl_kw)

    if is_main:
        info(f"Creating model of type: {model_type}")
    model_def['pressure_names'] = dataset.PRESSURE_NAMES
    model_def['evaptrans_names'] = dataset.EVAPTRANS_NAMES
    model_def['param_names'] = dataset.PARAM_NAMES
    model_def['n_evaptrans'] = dataset.n_evaptrans
    model_def['parameter_list'] = dataset.parameter_list
    model_def['param_nlayer'] = dataset.param_nlayer
    model = get_model(model_type, model_def)
    model = model.to(train_device).to(dtype)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    if is_main:
        verbose(f"Model created and moved to {train_device} with dtype {dtype}")

    if is_main:
        info(f"Setting up optimizer ({optimizer}) and loss function ({loss})")
    optimizer_obj = get_optimizer(optimizer, model, eff_lr)
    loss_fn = get_loss(loss)

    scheduler = None
    if 'callbacks' in config and 'lr_scheduler' in config['callbacks']:
        lr_config = config['callbacks']['lr_scheduler']
        if lr_config.get('enabled', False):
            scheduler_type = lr_config.get('type', 'ReduceLROnPlateau')
            scheduler = get_scheduler(scheduler_type, optimizer_obj, **lr_config)
            if is_main:
                info(f"Learning rate scheduler created: {scheduler_type}")

    autoregressive_loss_weights = None
    if 'autoregressive' in config and config['autoregressive'] is not None:
        autoregressive_config = config['autoregressive']
        autoregressive_loss_weights = autoregressive_config.get('loss_weights', None)
        if autoregressive_loss_weights and is_main:
            info(f"Using custom autoregressive loss weights: {autoregressive_loss_weights}")

    callback_manager = CallbackManager()
    callbacks = create_callbacks_from_config(config, model, log_location, name)
    for callback in callbacks:
        callback_manager.add_callback(callback)

    tensorboard_tracker = create_tensorboard_tracker_from_config(config, name)
    if tensorboard_tracker and is_main:
        callback_manager.add_callback(tensorboard_tracker)
        info("TensorBoard tracking enabled")

    try:
        if is_main:
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
            device=train_device,
            dtype=dtype,
            autoregressive_loss_weights=autoregressive_loss_weights,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            rank=rank,
        )
        if distributed:
            dist.barrier()

        if is_main:
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

            config["data_def"].pop("n_timesteps")
            with open(f'{log_location}/{name}_config.yaml', 'w') as f:
                yaml.safe_dump(config, f)

            verbose(f"Saving metrics to {metrics_filename}")
            metrics.to_csv(metrics_filename)

            to_save = model.module if distributed else model
            verbose(f"Saving model weights to {weights_filename}")
            torch.save(to_save.state_dict(), weights_filename)

            verbose(f"Creating and saving TorchScript model to {model_filename}")
            to_save.eval()
            m = torch.jit.script(to_save)
            torch.jit.save(m, model_filename)

            info("Training process completed successfully")
            print('----------------------------------------')
            print(f'Metrics saved to {metrics_filename}')
            print(f'Model saved to {model_filename}')
            print(f'Config saved to {log_location}/{name}_config.yaml')
    finally:
        if distributed:
            dist.destroy_process_group()



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
    test_data_def = data_def.copy()
    test_data_location = test_data_def.pop('test_data_location')
    test_data_def['data_location'] = test_data_location
    test_data_def['run_name'] = name
    dataset = ParFlowDataset(**test_data_def, dtype=dtype)
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
            s, e, p, y = batch

            if save_inputs:
                all_states.append(s)
                all_evaptrans.append(e)
                if i == 0:
                    all_parameters = p

            s = s.to(device=device, non_blocking=True)
            e = e.to(device=device, non_blocking=True)
            p = p.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            model.scale_pressure(s)
            model.scale_evaptrans(e)
            model.scale_statics(p)
            model.scale_pressure(y)

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

    info("Evaluation completed, processing results")
    if save_inputs:
        all_states = torch.cat(all_states)
        all_evaptrans = torch.cat(all_evaptrans)
        all_scaled_states = torch.cat(all_scaled_states)
        all_scaled_evaptrans = torch.cat(all_scaled_evaptrans)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
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
    metrics = calculate_metrics(all_outputs, all_targets)
    metrics_filename = f'{log_location}/{name}_test_metrics.csv'
    verbose(f"Saving metrics to {metrics_filename}")
    metrics.to_csv(metrics_filename)
    info(f'Test metrics saved to {metrics_filename}')
    info("Test results:")
    print(metrics)

    info("Testing process completed successfully")


def main(config, mode, log_level, save_inputs):
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


if __name__ == "__main__":
    # EXAMPLE USAGE: python main.py --config example_config.yaml --mode train --log-level info
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "test"], default="train"
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
    main(args.config, args.mode, args.log_level, save_inputs)
