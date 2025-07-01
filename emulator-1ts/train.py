import torch
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, Any
from logger import info, verbose, error, get_log_level, LogLevel
from utils import compute_channel_losses, compute_quantile_metrics
from callbacks import CallbackManager


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64

def train_epoch(
    model,
    dataset,
    optimizer,
    loss_fn,
    device,
    callback_manager: Optional[CallbackManager] = None,
    train=True,
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
    
    # Store sample predictions for visualization
    sample_predictions = None
    sample_targets = None
    
    for i, batch in enumerate(data_iterator):
        # Callback: batch begin
        if callback_manager:
            callback_manager.on_batch_begin(i, {'training': train})
        
        state, evaptrans, params, y = batch
        state = state.to(device=device, non_blocking=True)
        evaptrans = evaptrans.to(device, non_blocking=True)
        params = params.to(device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        y = y.squeeze()

        model.scale_pressure(state)
        model.scale_evaptrans(evaptrans)
        model.scale_statics(params)
        model.scale_pressure(y)

        if not len(state): 
            continue
            
        optimizer.zero_grad()
        if train:
            yhat = model(state, evaptrans, params).squeeze()
        else:
            # Don't compute gradients when in validation mode. Saves on computation
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
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # Store first batch for visualization (validation only)
        if not train and i == 0:
            # Unscale for visualization
            model.unscale_pressure(yhat)
            model.unscale_pressure(y)
            sample_predictions = yhat.detach().cpu()
            sample_targets = y.detach().cpu()
    
    avg_loss = epoch_loss / max(num_batches, 1)
    
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
    dtype=DTYPE
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
            device=device
        )
        train_df = train_df._append(train_metrics, ignore_index=True)
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
                device=device
            )
            valid_df = valid_df._append(valid_metrics, ignore_index=True)
            vl = valid_metrics['val_loss']
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
