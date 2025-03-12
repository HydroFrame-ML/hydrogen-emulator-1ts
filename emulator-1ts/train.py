import torch
import pandas as pd
from tqdm import tqdm
from logger import info, verbose, error, get_log_level, LogLevel


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

def train_epoch(
    model,
    dataset,
    optimizer,
    loss_fn,
    device,
    train=True,
):
    # Trains 1 epoch
    prefix = 'train' if train else 'valid'
    verbose(f"Starting {'training' if train else 'validation'} epoch")
    
    # Use tqdm progress bar only in verbose mode
    is_verbose = get_log_level() == LogLevel.VERBOSE
    
    # Wrap dataset with tqdm if in verbose mode
    data_iterator = tqdm(dataset, desc=f"{'Training' if train else 'Validation'} batch") if is_verbose else dataset
    
    for i, batch in enumerate(data_iterator):
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

        if not len(state): continue
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
            optimizer.step()
        
    return pd.Series({f'{prefix}_loss': loss.item()})

def train_model(
    model, 
    train_dl, 
    opt, 
    loss_fun, 
    max_epochs,
    scheduler=None,
    val_dl=None, 
    device=DEVICE, 
    dtype=DTYPE
):
    info(f"Starting model training for {max_epochs} epochs")
    verbose(f"Using device: {device}, dtype: {dtype}")
    train_df, valid_df = pd.DataFrame(), pd.DataFrame()
    for e in (bar := tqdm(range(max_epochs))):
        # Make sure to turn on train mode here
        # so that we update parameters
        model.train()
        train_metrics = train_epoch(model, train_dl, opt, loss_fun, train=True, device=device)
        train_df = train_df._append(train_metrics, ignore_index=True)
        tl = train_metrics['train_loss']
        info(f"Epoch {e+1}/{max_epochs} - Train loss: {tl:0.4e}")

        if val_dl is not None:
            # Now set to evaluation mode which reduces
            # the memory/computational cost
            model.eval()
            valid_metrics = train_epoch(model, val_dl, opt, loss_fun, train=False, device=device)
            valid_df = valid_df._append(valid_metrics, ignore_index=True)
            vl = valid_metrics['valid_loss']
            info(f"Epoch {e+1}/{max_epochs} - Validation loss: {vl:0.4e}")

            bar.set_description(f'Train loss: {tl:0.1e}, valid loss: {vl:0.1e}')
        else:
            bar.set_description(f'Train loss: {tl:0.1e}')

        if scheduler is not None:
            scheduler.step()

    if val_dl is not None:
        # Merge the two dataframes
        train_df = pd.concat([train_df, valid_df], axis=1)
    info("Training completed")
    return train_df
