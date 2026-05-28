import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, Any, Union
from .logger import info, verbose, error, get_log_level, LogLevel
from .utils import compute_channel_losses, compute_quantile_metrics
from .callbacks import CallbackManager


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64


def _sync_weighted_loss(weighted_sum: float, n_samples: float, device: torch.device) -> float:
    """All-reduce weighted loss sums for a correct global mean when using DDP."""
    if not dist.is_available() or not dist.is_initialized():
        return weighted_sum / max(n_samples, 1)
    t = torch.tensor([weighted_sum, n_samples], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / max(t[1], 1)).item()


def _batch_size_from_state(state: torch.Tensor) -> int:
    return int(state.shape[0])

def calculate_multistep_loss(predictions, targets, loss_fn, weights=None):
    """
    Calculate loss across multiple timesteps with optional weighting.
    
    Args:
        predictions: Model predictions [n_timesteps, batch, z, y, x]
        targets: Ground truth targets [n_timesteps, batch, z, y, x]
        loss_fn: Loss function to use
        weights: Optional weights for each timestep [n_timesteps]
        
    Returns:
        Weighted average loss across timesteps
    """
    n_timesteps = predictions.shape[0]
    
    if weights is None:
        # Default: equal weighting across all timesteps
        weights = torch.ones(n_timesteps, device=predictions.device)
    else:
        weights = torch.tensor(weights, device=predictions.device, dtype=predictions.dtype)
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    total_loss = 0.0
    for t in range(n_timesteps):
        timestep_loss = loss_fn(predictions[t], targets[t])
        total_loss += weights[t] * timestep_loss
    
    return total_loss

def train_epoch(
    model,
    dataset,
    optimizer,
    loss_fn,
    device: Union[str, torch.device],
    callback_manager: Optional[CallbackManager] = None,
    train=True,
    autoregressive_loss_weights=None,
    rank: int = 0,
):
    # Trains 1 epoch
    prefix = 'train' if train else 'val'
    device_t = torch.device(device) if not isinstance(device, torch.device) else device
    raw_model = model.module if hasattr(model, "module") else model
    is_main = rank == 0
    if is_main:
        verbose(f"Starting {'training' if train else 'validation'} epoch")
    # Use tqdm progress bar only in verbose mode on rank 0
    is_verbose = get_log_level() == LogLevel.VERBOSE and is_main
    data_iterator = tqdm(dataset, desc=f"{'Training' if train else 'Validation'} batch") if is_verbose else dataset

    weighted_loss = 0.0
    total_samples = 0.0
    
    # Store sample predictions for visualization
    sample_predictions = None
    sample_targets = None
    
    for i, batch in enumerate(data_iterator):
        # Callback: batch begin
        if callback_manager:
            callback_manager.on_batch_begin(i, {'training': train})
        
        state, evaptrans, params, y = batch
        state = state.to(device=device_t, non_blocking=True)
        evaptrans = evaptrans.to(device_t, non_blocking=True)
        params = params.to(device_t, non_blocking=True)
        y = y.to(device=device_t, non_blocking=True)
        
        # Detect multi-timestep vs single-timestep based on target shape
        is_multistep = len(y.shape) == 5  # [n_timesteps, batch, z, y, x]
        
        if is_multistep:
            # Multi-timestep autoregressive training
            n_timesteps = y.shape[0]
            
            # Scale data
            raw_model.scale_pressure(state)
            raw_model.scale_statics(params)
            # Scale evaptrans sequence
            for t in range(n_timesteps):
                raw_model.scale_evaptrans(evaptrans[t])
            # Scale target sequence
            for t in range(n_timesteps):
                raw_model.scale_pressure(y[t])
        else:
            # Single-timestep training (backward compatibility)
            y = y.squeeze()
            raw_model.scale_pressure(state)
            raw_model.scale_evaptrans(evaptrans)
            raw_model.scale_statics(params)
            raw_model.scale_pressure(y)

        if not len(state):
            continue

        batch_n = _batch_size_from_state(state)

        optimizer.zero_grad()
        
        if is_multistep:
            # Multi-timestep autoregressive prediction
            if train:
                yhat = raw_model.forward_autoregressive(state, evaptrans, params)
            else:
                with torch.no_grad():
                    yhat = raw_model.forward_autoregressive(state, evaptrans, params)
                    
            if torch.isnan(yhat).any():
                error(f"NaN values detected in predictions: {torch.isnan(yhat).sum()} NaNs")
                error(f"NaN values in input state: {torch.isnan(state).sum()} NaNs")
                raise ValueError(f'Predictions went nan! Nans in input: {torch.isnan(state).sum()}')
                
            # Calculate multi-timestep loss
            loss = calculate_multistep_loss(yhat, y, loss_fn, autoregressive_loss_weights)
        else:
            # Single-timestep prediction (backward compatibility)  
            if train:
                yhat = model(state, evaptrans, params).squeeze()
            else:
                with torch.no_grad():
                    yhat = model(state, evaptrans, params).squeeze()
                    
            if torch.isnan(yhat).any():
                error(f"NaN values detected in predictions: {torch.isnan(yhat).sum()} NaNs")
                error(f"NaN values in input state: {torch.isnan(state).sum()} NaNs")
                raise ValueError(f'Predictions went nan! Nans in input: {torch.isnan(state).sum()}')
                
            loss = loss_fn(yhat, y)
        
        if train:
            loss.backward()
            
            # Callback: batch end (for gradient clipping)
            if callback_manager:
                callback_manager.on_batch_end(i, {'training': train, 'loss': loss.item()})
            
            optimizer.step()
        else:
            # Callback: batch end
            if callback_manager:
                callback_manager.on_batch_end(i, {'training': train, 'loss': loss.item()})
        
        weighted_loss += loss.item() * batch_n
        total_samples += batch_n

        # Store first batch for visualization (validation only, rank 0)
        if not train and i == 0 and is_main:
            # Unscale for visualization
            if is_multistep:
                # Unscale sequence predictions and targets
                yhat_viz = yhat.clone()
                y_viz = y.clone()
                for t in range(n_timesteps):
                    raw_model.unscale_pressure(yhat_viz[t])
                    raw_model.unscale_pressure(y_viz[t])
                sample_predictions = yhat_viz.detach().cpu()
                sample_targets = y_viz.detach().cpu()
            else:
                # Single-timestep unscaling
                yhat_viz = yhat.clone()
                y_viz = y.clone()
                raw_model.unscale_pressure(yhat_viz)
                raw_model.unscale_pressure(y_viz)
                sample_predictions = yhat_viz.detach().cpu()
                sample_targets = y_viz.detach().cpu()
    
    avg_loss = _sync_weighted_loss(weighted_loss, total_samples, device_t)
    
    # Compute additional metrics for validation
    metrics = {f'{prefix}_loss': avg_loss}
    
    if not train and sample_predictions is not None and sample_targets is not None:
        # Compute channel losses
        channel_losses = compute_channel_losses(sample_predictions, sample_targets)
        metrics.update(channel_losses)
        
        # Compute quantile metrics
        quantile_metrics = compute_quantile_metrics(sample_predictions, sample_targets)
        metrics.update(quantile_metrics)
        
        # Add sample data for visualization
        metrics['sample_predictions'] = sample_predictions
        metrics['sample_targets'] = sample_targets
        
    return pd.Series(metrics)

def train_model(
    model,
    train_dl,
    opt,
    loss_fun,
    max_epochs,
    scheduler=None,
    val_dl=None,
    callback_manager: Optional[CallbackManager] = None,
    device=DEVICE,
    dtype=DTYPE,
    autoregressive_loss_weights=None,
    train_sampler=None,
    val_sampler=None,
    rank: int = 0,
):
    is_main = rank == 0
    if is_main:
        info(f"Starting model training for {max_epochs} epochs")
        verbose(f"Using device: {device}, dtype: {dtype}")
    
    # Initialize training logs
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    
    # Callback: training begin
    if callback_manager:
        callback_manager.on_train_begin({'model': model, 'optimizer': opt, 'scheduler': scheduler})
    
    epoch_range = range(max_epochs)
    bar = tqdm(epoch_range, disable=not is_main)
    for e in bar:
        if train_sampler is not None:
            train_sampler.set_epoch(e)
        if val_sampler is not None:
            val_sampler.set_epoch(e)

        # Callback: epoch begin
        if callback_manager:
            callback_manager.on_epoch_begin(e, {'epoch': e})

        # Make sure to turn on train mode here
        # so that we update parameters
        model.train()
        train_metrics = train_epoch(
            model,
            train_dl,
            opt,
            loss_fun,
            callback_manager=callback_manager,
            train=True,
            device=device,
            autoregressive_loss_weights=autoregressive_loss_weights,
            rank=rank,
        )
        train_df = train_df._append(train_metrics, ignore_index=True)
        tl = train_metrics['train_loss']
        if is_main:
            info(f"Epoch {e+1}/{max_epochs} - Train loss: {tl:0.4e}")

        # Prepare epoch logs
        epoch_logs = {'train_loss': tl, 'epoch': e}
        
        if val_dl is not None:
            # Now set to evaluation mode which reduces
            # the memory/computational cost
            model.eval()
            valid_metrics = train_epoch(
                model,
                val_dl,
                opt,
                loss_fun,
                callback_manager=callback_manager,
                train=False,
                device=device,
                autoregressive_loss_weights=autoregressive_loss_weights,
                rank=rank,
            )
            valid_df = valid_df._append(valid_metrics, ignore_index=True)
            vl = valid_metrics['val_loss']
            if is_main:
                info(f"Epoch {e+1}/{max_epochs} - Validation loss: {vl:0.4e}")

            # Add validation metrics to epoch logs
            epoch_logs['val_loss'] = vl
            
            # Add channel losses and quantile metrics if available
            for key, value in valid_metrics.items():
                if key.startswith(('loss_channel_', 'error_q')) and isinstance(value, (int, float)):
                    epoch_logs[key] = value
            
            # Add sample data for visualization
            if 'sample_predictions' in valid_metrics:
                epoch_logs['sample_predictions'] = valid_metrics['sample_predictions']
            if 'sample_targets' in valid_metrics:
                epoch_logs['sample_targets'] = valid_metrics['sample_targets']

            bar.set_description(f'Train loss: {tl:0.1e}, val loss: {vl:0.1e}')
        else:
            bar.set_description(f'Train loss: {tl:0.1e}')

        # Learning rate scheduling
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                if 'val_loss' in epoch_logs:
                    # For ReduceLROnPlateau
                    try:
                        # Ensure val_loss is a float
                        val_loss_value = float(epoch_logs['val_loss'])
                        scheduler.step(val_loss_value)
                    except (TypeError, ValueError) as ex:
                        print('Factor: ', scheduler.factor, type(scheduler.factor))
                        print('Min LR: ', scheduler.min_lrs[-1], type(scheduler.min_lrs[-1]))
                        print('Val loss: ', epoch_logs['val_loss'], type(epoch_logs['val_loss']))

                        # For other schedulers that don't take metrics, or if conversion fails
                        error(f"Scheduler step failed with metrics: {ex}")
                        scheduler.step()
                else: # Step on train loss
                    try:
                        # Ensure train_loss is a float
                        train_loss_value = float(epoch_logs['train_loss'])
                        scheduler.step(train_loss_value)
                    except (TypeError, ValueError) as ex:
                        error(f"Scheduler step failed with metrics: {ex}")
                        scheduler.step()
        
        # Log learning rate and optimizer state (always log, even without scheduler)
        if hasattr(scheduler, 'get_last_lr') and scheduler is not None:
            epoch_logs['learning_rate'] = scheduler.get_last_lr()[0]
        elif hasattr(opt, 'param_groups'):
            epoch_logs['learning_rate'] = opt.param_groups[0]['lr']
        
        # Callback: epoch end
        if callback_manager:
            callback_manager.on_epoch_end(e, epoch_logs)
            
            # Check for early stopping
            if epoch_logs.get('stop_training', False):
                info(f"Training stopped early at epoch {e+1}")
                break

    # Callback: training end
    if callback_manager:
        final_logs = {'train_df': train_df}
        if val_dl is not None:
            final_logs['valid_df'] = valid_df
        callback_manager.on_train_end(final_logs)

    if val_dl is not None:
        train_df['val_loss'] = valid_df['val_loss']
    if is_main:
        info("Training completed")
    return train_df
