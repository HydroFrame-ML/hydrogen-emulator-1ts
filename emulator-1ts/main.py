import yaml
import torch
import pandas as pd

from dataset import ParFlowDataset
from model import get_model
from train import train_model
from argparse import ArgumentParser
from utils import get_optimizer, get_loss, get_dtype, calculate_metrics
from torch.utils.data import DataLoader

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
    # Create the data loader
    dtype = get_dtype(dtype)
    dataset = ParFlowDataset(**data_def, dtype=dtype)
    train_dl = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=custom_collate, 
        shuffle=True, 
        num_workers=num_workers
    )

    # Create the model
    # Add names of model inputs to model definition for scaling, if needed
    model_def['pressure_names'] = dataset.PRESSURE_NAMES
    model_def['evaptrans_names'] = dataset.EVAPTRANS_NAMES
    model_def['param_names'] = dataset.PARAM_NAMES
    model_def['n_evaptrans'] = dataset.n_evaptrans
    model_def['parameter_list'] = dataset.parameter_list
    model_def['param_nlayer'] = dataset.param_nlayer
    model = get_model(model_type, model_def)
    model = model.to(device).to(dtype)


    # Create the optimizer and loss function
    optimizer = get_optimizer(optimizer, model, lr)
    loss_fn = get_loss(loss)

    metrics = train_model(
        model, train_dl, optimizer, loss_fn, n_epochs, device=device
    )
    print('----------------------------------------')
    print(metrics)
    print('----------------------------------------')
    
    metrics_filename = f'{log_location}/{name}_metrics.csv'
    weights_filename = f'{log_location}/{name}_weights_only.pth'
    model_filename = f'{log_location}/{name}_model.pth'
    config['model_path'] = model_filename
    config['weights_path'] = weights_filename
    config['metrics_path'] = metrics_filename
    with open(f'{log_location}/{name}_config.yaml', 'w') as f:
        yaml.safe_dump(config, f)
    metrics.to_csv(metrics_filename)
    model = model.to(torch.float64)
    torch.save(model.state_dict(), weights_filename)
    m = torch.jit.script(model)
    torch.jit.save(m, model_filename)

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
    dtype = get_dtype(dtype)
    # Load the model
    model = torch.jit.load(model_path)
    model = model.to(device).to(dtype)
    
    # Create the data loader
    dataset = ParFlowDataset(**data_def, dtype=dtype)
    test_dl = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=custom_collate, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dl:
            s, e, p, y = batch
            s, e, p, y = s.to(device), e.to(device), p.to(device), y.to(device)
            outputs = model(s, e, p)
            all_outputs.append(outputs.cpu())
            all_targets.append(y.cpu())
    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    print(f'All outputs shape: {all_outputs.shape}')
    print(f'All targets shape: {all_targets.shape}')
    
    # Save the outputs
    output_filename = f'{log_location}/{name}_outputs.pt'
    torch.save(all_outputs, output_filename)
    print(f'Outputs saved to {output_filename}')
    
    # Calculate and print metrics
    metrics = calculate_metrics(all_outputs, all_targets)
    metrics_filename = f'{log_location}/{name}_test_metrics.csv'
    metrics.to_csv(metrics_filename)
    print(f'Test metrics saved to {metrics_filename}')
    print(metrics)


def main(config, mode):
    config = read_config(config)

    if mode == "train":
        print("TRAINING")
        train(**config, config=config)
    elif mode == "test":
        print("TESTING")
        test(**config)


if __name__ == "__main__":
    # EXAMPLE USAGE: python main.py --config example_config.yaml --mode train
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "test"], default="train"
    )
    args = parser.parse_args()
    main(args.config, args.mode)
