import yaml
import torch
import pandas as pd
from tqdm import tqdm

from dataset import ParFlowDataset
from model import get_model
from train import train_model
from argparse import ArgumentParser
from utils import get_optimizer, get_loss, get_dtype, calculate_metrics
from torch.utils.data import DataLoader
from logger import set_log_level, info, verbose, error, LogLevel, get_log_level

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
    e = torch.stack(e)
    p = torch.stack(p)
    y = torch.stack(y)
    return s, e, p, y

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
    config: dict,
    **kwargs
):
    info(f"Initializing training with name: {name}")
    verbose(f"Training parameters: epochs={n_epochs}, batch_size={batch_size}, lr={lr}, device={device}")
    
    # Create the data loaders
    dtype = get_dtype(dtype)
    info("Creating training dataset and data loader")
    train_data_def = data_def.copy()
    train_data_location = train_data_def.pop('train_data_location')
    train_data_def['data_location'] = train_data_location
    train_data_def['run_name'] = name
    dataset = ParFlowDataset(**train_data_def, dtype=dtype)
    verbose(f"Training dataset created with {len(dataset)} samples")
    train_dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate,
        num_workers=num_workers
    )

    val_dl = None
    if 'validation_data_location' in data_def:
        info("Creating validation dataset and data loader")
        validation_data_def = data_def.copy()
        validation_data_location = validation_data_def.pop('validation_data_location')
        validation_data_def['data_location'] = validation_data_location
        validation_data_def['run_name'] = name
        val_dataset = ParFlowDataset(**validation_data_def, dtype=dtype)
        verbose(f"Validation dataset created with {len(val_dataset)} samples")
        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=custom_collate,
            num_workers=num_workers
        )

    # Create the model
    info(f"Creating model of type: {model_type}")
    # Add names of model inputs to model definition for scaling, if needed
    model_def['pressure_names'] = dataset.PRESSURE_NAMES
    model_def['evaptrans_names'] = dataset.EVAPTRANS_NAMES
    model_def['param_names'] = dataset.PARAM_NAMES
    model_def['n_evaptrans'] = dataset.n_evaptrans
    model_def['parameter_list'] = dataset.parameter_list
    model_def['param_nlayer'] = dataset.param_nlayer
    model = get_model(model_type, model_def)
    model = model.to(device).to(dtype)
    verbose(f"Model created and moved to {device} with dtype {dtype}")


    # Create the optimizer and loss function
    info(f"Setting up optimizer ({optimizer}) and loss function ({loss})")
    optimizer = get_optimizer(optimizer, model, lr)
    loss_fn = get_loss(loss)

    info("Starting model training")
    metrics = train_model(model, train_dl, optimizer, loss_fn, n_epochs, val_dl=val_dl, device=device)
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
    with open(f'{log_location}/{name}_config.yaml', 'w') as f:
        yaml.safe_dump(config, f)

    verbose(f"Saving metrics to {metrics_filename}")
    metrics.to_csv(metrics_filename)

    verbose("Converting model to float64 for saving")
    model = model.to(torch.float64)

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



def test(
    name: str,
    log_location: str,
    model_path: str,
    data_def: dict,
    batch_size: int,
    device: str,
    num_workers: int,
    dtype: str,
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
    
    verbose("Processing test batches")
    
    # Use tqdm progress bar only in verbose mode
    is_verbose = get_log_level() == LogLevel.VERBOSE
    
    # Wrap test_dl with tqdm if in verbose mode
    batch_iterator = tqdm(test_dl, desc="Testing batch") if is_verbose else test_dl
    
    with torch.no_grad():
        for i, batch in enumerate(batch_iterator):
            s, e, p, y = batch
            s = s.to(device=device, non_blocking=True)
            e = e.to(device=device, non_blocking=True)
            p = p.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            model.scale_pressure(s)
            model.scale_evaptrans(e)
            model.scale_statics(p)
            model.scale_pressure(y)

            outputs = model(s, e, p)

            # Unscale the outputs
            model.unscale_pressure(outputs)
            model.unscale_pressure(y)

            all_outputs.append(outputs.cpu())
            all_targets.append(y.cpu())
    
    info("Evaluation completed, processing results")
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


def main(config, mode, log_level):
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
        test(**config)


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
    args = parser.parse_args()
    main(args.config, args.mode, args.log_level)
