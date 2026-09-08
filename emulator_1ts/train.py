import torch
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, Any
from .logger import info, verbose, error, get_log_level, LogLevel
from .losses import StructuredLoss
from .utils import (
    compute_boundary_losses,
    compute_channel_losses,
    compute_quantile_metrics,
    persistence_skill_sums,
    ponding_flip_sums,
    pressure_layer_sigmas,
    pressure_top_zero_scaled,
    skill_from_sums,
)
from .callbacks import CallbackManager


DTYPE = torch.float64


def calculate_masked_loss(predictions, targets, valid_mask, loss_fn):
    """Apply a loss only to valid ParFlow cells.

    Structured losses need the spatial layout to difference neighbouring
    layers, so they receive the full tensors and do their own masking.
    Pointwise losses keep the original flattened path.
    """

    valid_mask = valid_mask.to(torch.bool)
    if not torch.any(valid_mask):
        raise ValueError("Cannot calculate loss: batch contains no valid cells")
    if isinstance(loss_fn, StructuredLoss):
        return loss_fn(predictions, targets, valid_mask)
    return loss_fn(predictions[valid_mask], targets[valid_mask])


def calculate_multistep_loss(
    predictions, targets, valid_masks, loss_fn, weights=None
):
    """
    Calculate loss across multiple timesteps with optional weighting.
    
    Args:
        predictions: Model predictions [n_timesteps, batch, z, y, x]
        targets: Ground truth targets [n_timesteps, batch, z, y, x]
        valid_masks: Valid target cells [n_timesteps, batch, z, y, x]
        loss_fn: Loss function to use
        weights: Optional weights for each timestep [n_timesteps]
        
    Returns:
        Weighted average loss across timesteps
    """
    n_timesteps = predictions.shape[0]
    
    if weights is None:
        # Default: equal weighting across all timesteps
        weights = torch.ones(
            n_timesteps, device=predictions.device, dtype=predictions.dtype
        )
    else:
        weights = torch.tensor(weights, device=predictions.device, dtype=predictions.dtype)
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    total_loss = 0.0
    for t in range(n_timesteps):
        timestep_loss = calculate_masked_loss(
            predictions[t], targets[t], valid_masks[t], loss_fn
        )
        total_loss += weights[t] * timestep_loss
    
    return total_loss


def forward_autoregressive_masked(
    model, initial_pressure, evaptrans_sequence, statics, valid_masks
):
    """Roll forward while keeping inactive cells neutral between steps."""

    predictions = []
    current_state = initial_pressure
    for t in range(evaptrans_sequence.shape[0]):
        next_state = model(current_state, evaptrans_sequence[t], statics)
        next_state = torch.where(
            valid_masks[t], next_state, torch.zeros_like(next_state)
        )
        predictions.append(next_state)
        current_state = next_state
    return torch.stack(predictions)

def train_epoch(
    model,
    dataset,
    optimizer,
    loss_fn,
    device,
    callback_manager: Optional[CallbackManager] = None,
    train=True,
    autoregressive_loss_weights=None,
):
    # Trains 1 epoch
    prefix = 'train' if train else 'val'
    verbose(f"Starting {'training' if train else 'validation'} epoch")
    
    # Use tqdm progress bar only in verbose mode
    is_verbose = get_log_level() == LogLevel.VERBOSE
    
    # Wrap dataset with tqdm if in verbose mode
    data_iterator = tqdm(dataset, desc=f"{'Training' if train else 'Validation'} batch") if is_verbose else dataset
    
    epoch_loss = 0.0
    num_batches = 0

    # Skill is accumulated over every batch rather than a single sample, since
    # it is the metric that decides whether a checkpoint is worth benchmarking.
    try:
        layer_sigmas = pressure_layer_sigmas(model)
    except (AttributeError, ValueError):
        layer_sigmas = None
    # Ponding flips are counted at the one-step horizon only: that is the
    # guess ParFlow consumes, and every flip lands the solver on the wrong
    # overland-flow branch.  Persistence's count is the acceptance bar.
    try:
        top_zero_scaled = pressure_top_zero_scaled(model)
    except (AttributeError, ValueError):
        top_zero_scaled = None
    skill_model_sse = 0.0
    skill_persistence_sse = 0.0
    skill_step0_model_sse = 0.0
    skill_step0_persistence_sse = 0.0
    pond_model_flips = 0.0
    pond_persistence_flips = 0.0
    pond_valid_cells = 0.0

    # Store sample predictions for visualization
    sample_predictions = None
    sample_targets = None
    sample_valid_mask = None
    
    for i, batch in enumerate(data_iterator):
        # Callback: batch begin
        if callback_manager:
            callback_manager.on_batch_begin(i, {'training': train})
        
        state, evaptrans, params, y, valid_mask = batch
        state = state.to(device=device, non_blocking=True)
        evaptrans = evaptrans.to(device, non_blocking=True)
        params = params.to(device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        valid_mask = valid_mask.to(device=device, non_blocking=True)
        
        # Detect multi-timestep vs single-timestep based on target shape
        is_multistep = len(y.shape) == 5  # [n_timesteps, batch, z, y, x]
        
        if is_multistep:
            # Multi-timestep autoregressive training
            n_timesteps = y.shape[0]
            
            # Scale data
            model.scale_pressure(state)
            model.scale_statics(params)
            # Scale evaptrans sequence
            for t in range(n_timesteps):
                model.scale_evaptrans(evaptrans[t])
            # Scale target sequence
            for t in range(n_timesteps):
                model.scale_pressure(y[t])
                y[t].masked_fill_(~valid_mask[t], 0)
            state.masked_fill_(~valid_mask[0], 0)
        else:
            # Single-timestep training (backward compatibility)
            model.scale_pressure(state)
            model.scale_evaptrans(evaptrans)
            model.scale_statics(params)
            model.scale_pressure(y)
            state.masked_fill_(~valid_mask, 0)
            y.masked_fill_(~valid_mask, 0)

        if not len(state): 
            continue
            
        optimizer.zero_grad()
        
        if is_multistep:
            # Multi-timestep autoregressive prediction
            if train:
                yhat = forward_autoregressive_masked(
                    model, state, evaptrans, params, valid_mask
                )
            else:
                with torch.no_grad():
                    yhat = forward_autoregressive_masked(
                        model, state, evaptrans, params, valid_mask
                    )
                    
            if torch.isnan(yhat).any():
                error(f"NaN values detected in predictions: {torch.isnan(yhat).sum()} NaNs")
                error(f"NaN values in input state: {torch.isnan(state).sum()} NaNs")
                raise ValueError(f'Predictions went nan! Nans in input: {torch.isnan(state).sum()}')
                
            # Calculate multi-timestep loss
            loss = calculate_multistep_loss(
                yhat, y, valid_mask, loss_fn, autoregressive_loss_weights
            )
        else:
            # Single-timestep prediction (backward compatibility)  
            if train:
                yhat = model(state, evaptrans, params)
            else:
                with torch.no_grad():
                    yhat = model(state, evaptrans, params)
                    
            if torch.isnan(yhat).any():
                error(f"NaN values detected in predictions: {torch.isnan(yhat).sum()} NaNs")
                error(f"NaN values in input state: {torch.isnan(state).sum()} NaNs")
                raise ValueError(f'Predictions went nan! Nans in input: {torch.isnan(state).sum()}')
                
            loss = calculate_masked_loss(yhat, y, valid_mask, loss_fn)
        
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
        
        epoch_loss += loss.item()
        num_batches += 1

        # Persistence for a step is the state the solver would otherwise reuse:
        # the input pressure for the first step, and the previous true state
        # afterwards.  Comparing against truth rather than the model's own
        # rollout keeps this a like-for-like baseline at every step.
        with torch.no_grad():
            if is_multistep:
                for t in range(n_timesteps):
                    reference = state if t == 0 else y[t - 1]
                    model_sse, persistence_sse = persistence_skill_sums(
                        yhat[t], y[t], reference, valid_mask[t], layer_sigmas
                    )
                    skill_model_sse += model_sse
                    skill_persistence_sse += persistence_sse
                    if t == 0:
                        skill_step0_model_sse += model_sse
                        skill_step0_persistence_sse += persistence_sse
                        if top_zero_scaled is not None:
                            flips, ref_flips, cells = ponding_flip_sums(
                                yhat[0], y[0], state, valid_mask[0],
                                top_zero_scaled,
                            )
                            pond_model_flips += flips
                            pond_persistence_flips += ref_flips
                            pond_valid_cells += cells
            else:
                model_sse, persistence_sse = persistence_skill_sums(
                    yhat, y, state, valid_mask, layer_sigmas
                )
                skill_model_sse += model_sse
                skill_persistence_sse += persistence_sse
                skill_step0_model_sse += model_sse
                skill_step0_persistence_sse += persistence_sse
                if top_zero_scaled is not None:
                    flips, ref_flips, cells = ponding_flip_sums(
                        yhat, y, state, valid_mask, top_zero_scaled
                    )
                    pond_model_flips += flips
                    pond_persistence_flips += ref_flips
                    pond_valid_cells += cells


        # Store first batch for visualization (validation only)
        if not train and i == 0:
            # Unscale for visualization
            if is_multistep:
                # Unscale sequence predictions and targets
                yhat_viz = yhat.clone()
                y_viz = y.clone()
                for t in range(n_timesteps):
                    model.unscale_pressure(yhat_viz[t])
                    model.unscale_pressure(y_viz[t])
                sample_predictions = yhat_viz.detach().cpu()
                sample_targets = y_viz.detach().cpu()
                sample_valid_mask = valid_mask.detach().cpu()
            else:
                # Single-timestep unscaling
                yhat_viz = yhat.clone()
                y_viz = y.clone()
                model.unscale_pressure(yhat_viz)
                model.unscale_pressure(y_viz)
                sample_predictions = yhat_viz.detach().cpu()
                sample_targets = y_viz.detach().cpu()
                sample_valid_mask = valid_mask.detach().cpu()
    
    avg_loss = epoch_loss / max(num_batches, 1)
    
    # Compute additional metrics for validation
    metrics = {f'{prefix}_loss': avg_loss}

    # 1.0 is perfect, 0.0 is no better than persistence, negative is worse than
    # doing nothing.  step0 is the one-step guess ParFlow actually consumes.
    metrics[f'{prefix}_skill'] = skill_from_sums(
        skill_model_sse, skill_persistence_sse
    )
    metrics[f'{prefix}_skill_step0'] = skill_from_sums(
        skill_step0_model_sse, skill_step0_persistence_sse
    )

    # One-step ponding flips per valid top-layer cell.  The model must reach
    # or beat the persistence rate before a checkpoint is worth benchmarking;
    # the ratio makes that a single number (<= 1.0 passes).
    if pond_valid_cells > 0:
        metrics[f'{prefix}_pond_flip_rate'] = (
            pond_model_flips / pond_valid_cells
        )
        metrics[f'{prefix}_pond_flip_rate_persistence'] = (
            pond_persistence_flips / pond_valid_cells
        )
        metrics[f'{prefix}_pond_flip_ratio'] = (
            pond_model_flips / pond_persistence_flips
            if pond_persistence_flips > 0
            else float('nan')
        )


    if not train and sample_predictions is not None and sample_targets is not None:
        # Compute channel losses
        channel_losses = compute_channel_losses(
            sample_predictions, sample_targets, mask=sample_valid_mask
        )
        metrics.update(channel_losses)

        # Track whether errors are concentrated near the masked domain edge,
        # where padding choices have the largest effect.
        boundary_losses = compute_boundary_losses(
            sample_predictions, sample_targets, mask=sample_valid_mask
        )
        metrics.update(boundary_losses)
        
        # Compute quantile metrics
        quantile_metrics = compute_quantile_metrics(
            sample_predictions, sample_targets, mask=sample_valid_mask
        )
        metrics.update(quantile_metrics)
        
        # Add sample data for visualization
        metrics['sample_predictions'] = sample_predictions
        metrics['sample_targets'] = sample_targets
        metrics['sample_valid_mask'] = sample_valid_mask
        
    return pd.Series(metrics)

def train_model(
    model, 
    train_dl, 
    opt, 
    loss_fun, 
    max_epochs,
    device,
    scheduler=None,
    val_dl=None, 
    callback_manager: Optional[CallbackManager] = None,
    dtype=DTYPE,
    autoregressive_loss_weights=None
):
    info(f"Starting model training for {max_epochs} epochs")
    verbose(f"Using device: {device}, dtype: {dtype}")
    
    # Initialize training logs
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    
    # Callback: training begin
    if callback_manager:
        callback_manager.on_train_begin({'model': model, 'optimizer': opt, 'scheduler': scheduler})
    
    for e in (bar := tqdm(range(max_epochs))):
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
            autoregressive_loss_weights=autoregressive_loss_weights
        )
        train_df = pd.concat(
            [train_df, train_metrics.to_frame().T], ignore_index=True
        )
        tl = train_metrics['train_loss']
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
                autoregressive_loss_weights=autoregressive_loss_weights
            )
            valid_df = pd.concat(
                [valid_df, valid_metrics.to_frame().T], ignore_index=True
            )
            vl = valid_metrics['val_loss']
            info(f"Epoch {e+1}/{max_epochs} - Validation loss: {vl:0.4e}")

            # Add validation metrics to epoch logs
            epoch_logs['val_loss'] = vl
            
            # Add channel losses and quantile metrics if available
            for key, value in valid_metrics.items():
                if key.startswith(
                    ('loss_channel_', 'loss_boundary', 'loss_interior', 'error_q')
                ) and isinstance(value, (int, float)):
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
    info("Training completed")
    return train_df
