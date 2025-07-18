import dask
import torch
import torch.nn.functional as F
import pandas as pd
from typing import Dict, List


dask.config.set(**{'array.slicing.split_large_chunks': True})


def calculate_metrics(outputs, targets):
    """Calculate comprehensive metrics for model evaluation."""
    metrics = {}
    
    # Basic metrics
    mse = torch.mean((outputs - targets) ** 2).item()
    mae = torch.mean(torch.abs(outputs - targets)).item()
    rmse = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
    
    metrics['MSE'] = mse
    metrics['MAE'] = mae
    metrics['RMSE'] = rmse
    
    # R-squared
    ss_res = torch.sum((targets - outputs) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    metrics['R2'] = r2.item()
    
    # Channel-wise metrics if 4D tensors
    if outputs.dim() == 4 and targets.dim() == 4:
        channel_metrics = compute_channel_losses(outputs, targets)
        metrics.update(channel_metrics)
    
    # Quantile metrics
    quantile_metrics = compute_quantile_metrics(outputs, targets)
    metrics.update(quantile_metrics)
    
    return pd.DataFrame([metrics])


def compute_channel_losses(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute loss for each channel separately.
    
    Args:
        predictions: Model predictions [batch, channels, height, width]
        targets: Ground truth [batch, channels, height, width]
    
    Returns:
        Dictionary with channel losses
    """
    channel_losses = {}
    
    if predictions.dim() == 4 and targets.dim() == 4:
        n_channels = predictions.shape[1]
        
        for i in range(n_channels):
            channel_loss = F.mse_loss(predictions[:, i], targets[:, i])
            channel_losses[f'loss_channel_{i}'] = channel_loss.item()
    
    return channel_losses


def compute_quantile_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                           quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]) -> Dict[str, float]:
    """
    Compute quantile metrics of prediction errors.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        quantiles: List of quantiles to compute
    
    Returns:
        Dictionary with quantile metrics
    """
    errors = torch.abs(predictions - targets)
    quantile_metrics = {}
    
    for q in quantiles:
        try:
            quantile_value = torch.quantile(errors, q)
            quantile_metrics[f'error_q{int(q*100)}'] = quantile_value.item()
        except Exception:
            # Fallback for older PyTorch versions
            errors_flat = errors.flatten().sort()[0]
            idx = int(q * len(errors_flat))
            quantile_value = errors_flat[idx]
            quantile_metrics[f'error_q{int(q*100)}'] = quantile_value.item()
    
    return quantile_metrics


def get_scheduler(scheduler_type: str, optimizer, **kwargs):
    """Get learning rate scheduler."""
    if scheduler_type == "ReduceLROnPlateau":
        # Convert min_lr to min_lrs (list) if provided as single value
        min_lr = float(kwargs.get('min_lr', 1e-7))
        if isinstance(min_lr, (int, float)):
            min_lrs = [min_lr] * len(optimizer.param_groups)
        else:
            min_lrs = min_lr
        
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=min_lrs
        )
    elif scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 1e-7)
        )
    else:
        raise ValueError(f"Scheduler {scheduler_type} not supported")


def get_dtype(dtype):
    if dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Data type {dtype} not supported")

def get_optimizer(optimizer_type, model, learning_rate, **kwargs):
    if optimizer_type == "adam":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.025, **kwargs
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, **kwargs
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")
    return optimizer

def get_loss(loss_type):
    if loss_type == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss_type == "mae":
        loss_fn = torch.nn.L1Loss()
    else:
        raise ValueError(f"Loss {loss_type} not supported")
    return loss_fn
