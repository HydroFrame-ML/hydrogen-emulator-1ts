import dask
import torch
import torch.nn.functional as F

import os
import json
import torch
import tempfile
import pandas as pd
import mlflow
from glob import glob
from tqdm.autonotebook import tqdm
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import TQDMProgressBar


dask.config.set(**{'array.slicing.split_large_chunks': True})


def calculate_metrics(outputs, targets):
    # Example metric calculation (mean squared error)
    mse = torch.mean((outputs - targets) ** 2).item()
    return pd.DataFrame({'MSE': [mse]})


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