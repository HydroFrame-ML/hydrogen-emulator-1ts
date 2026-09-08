import dask
import torch
import torch.nn.functional as F
import pandas as pd
from typing import Dict, List, Optional

from .losses import PressureFirstGuessLoss, StructuredLoss
from .scalers import resolve_pressure_scaler_names


dask.config.set(**{'array.slicing.split_large_chunks': True})


def _validated_mask(reference: torch.Tensor, mask: Optional[torch.Tensor]):
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if mask.shape != reference.shape:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} does not match tensor shape "
            f"{tuple(reference.shape)}"
        )
    return mask.to(device=reference.device, dtype=torch.bool)


def persistence_skill_sums(
    predictions, targets, persistence, mask=None, layer_sigmas=None
):
    """Squared-error sums for the model and for a persistence baseline.

    Skill is the metric that predicts whether a first guess helps ParFlow, and
    MSE demonstrably is not: a model reached a pressure RMSE of 1.3e-3 while
    reducing the distance to the next-step solution by only 1.8%, and it lost
    every gated timestep.  Persistence is the guess ParFlow already makes, so a
    model with skill <= 0 cannot accelerate the solver no matter how good its
    loss looks.

    Args:
        predictions: Model output, ``(batch, z, y, x)``.
        targets: True next state, same shape.
        persistence: The previous state, i.e. what the solver would have used.
        mask: Optional bool tensor of valid cells.
        layer_sigmas: Per-layer sigmas.  Supply these when the tensors are in
            scaled units so the ratio is formed on physical errors, which
            weights layers the way the solver sees them.

    Returns:
        ``(model_sse, persistence_sse)`` as floats.
    """
    valid_mask = _validated_mask(targets, mask)
    if not torch.any(valid_mask):
        return 0.0, 0.0

    model_error = predictions - targets
    persistence_error = persistence - targets

    if layer_sigmas is not None:
        channel_axis = 1 if predictions.dim() == 4 else 2
        sigmas = torch.as_tensor(
            layer_sigmas, device=predictions.device, dtype=predictions.dtype
        )
        if sigmas.numel() != predictions.shape[channel_axis]:
            raise ValueError(
                f"layer_sigmas has {sigmas.numel()} entries but the prediction "
                f"has {predictions.shape[channel_axis]} layers"
            )
        shape = [1] * predictions.dim()
        shape[channel_axis] = -1
        sigmas = sigmas.view(shape)
        model_error = model_error * sigmas
        persistence_error = persistence_error * sigmas

    return (
        float(torch.sum(model_error[valid_mask] ** 2)),
        float(torch.sum(persistence_error[valid_mask] ** 2)),
    )


def skill_from_sums(model_sse, persistence_sse):
    """Convert squared-error sums into a persistence skill score.

    1.0 is a perfect prediction, 0.0 is no better than persistence, and
    negative values are worse than doing nothing.
    """
    if persistence_sse <= 0.0:
        return float("nan")
    return 1.0 - (model_sse / persistence_sse) ** 0.5


def calculate_metrics(outputs, targets, mask=None):
    """Calculate comprehensive metrics for model evaluation."""
    metrics = {}
    valid_mask = _validated_mask(targets, mask)
    if not torch.any(valid_mask):
        raise ValueError("Cannot calculate metrics: no valid cells")
    valid_outputs = outputs[valid_mask]
    valid_targets = targets[valid_mask]

    # Basic metrics
    mse = torch.mean((valid_outputs - valid_targets) ** 2).item()
    mae = torch.mean(torch.abs(valid_outputs - valid_targets)).item()
    rmse = torch.sqrt(torch.mean((valid_outputs - valid_targets) ** 2)).item()
    
    metrics['MSE'] = mse
    metrics['MAE'] = mae
    metrics['RMSE'] = rmse
    
    # R-squared
    ss_res = torch.sum((valid_targets - valid_outputs) ** 2)
    ss_tot = torch.sum((valid_targets - torch.mean(valid_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    metrics['R2'] = r2.item()
    
    # Channel-wise metrics if 4D tensors
    if outputs.dim() in (4, 5) and targets.dim() == outputs.dim():
        channel_metrics = compute_channel_losses(outputs, targets, mask=valid_mask)
        metrics.update(channel_metrics)
        boundary_metrics = compute_boundary_losses(
            outputs, targets, mask=valid_mask
        )
        metrics.update(boundary_metrics)
    
    # Quantile metrics
    quantile_metrics = compute_quantile_metrics(
        outputs, targets, mask=valid_mask
    )
    metrics.update(quantile_metrics)
    
    return pd.DataFrame([metrics])


def compute_boundary_losses(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    boundary_width: int = 5,
) -> Dict[str, float]:
    """Compare MSE near the domain edge with MSE in its interior.

    ``boundary_width=5`` matches the receptive-field radius of the current
    five-block, 3x3 MJB ConvNeXT. Cells outside the rectangular tensor are
    treated as inactive, so this diagnostic covers both the irregular domain
    edge and the outer tensor edge.
    """

    if predictions.dim() not in (4, 5) or targets.dim() != predictions.dim():
        return {}
    if boundary_width < 1:
        raise ValueError("boundary_width must be at least 1")

    valid_mask = _validated_mask(targets, mask)
    if predictions.dim() == 4:
        flat_predictions = predictions
        flat_targets = targets
        flat_mask = valid_mask
    else:
        # Treat each timestep/batch pair as an independent spatial sample.
        flat_predictions = predictions.flatten(0, 1)
        flat_targets = targets.flatten(0, 1)
        flat_mask = valid_mask.flatten(0, 1)

    # Erode each output channel independently in case vertical validity differs.
    interior = flat_mask.reshape(-1, 1, *flat_mask.shape[-2:])
    kernel = torch.ones(
        (1, 1, 3, 3), device=predictions.device, dtype=predictions.dtype
    )
    for _ in range(boundary_width):
        neighbor_count = F.conv2d(
            interior.to(predictions.dtype), kernel, padding=1
        )
        interior = interior & (neighbor_count == 9)

    boundary = flat_mask & ~interior.reshape_as(flat_mask)
    interior = interior.reshape_as(flat_mask) & flat_mask

    squared_error = (flat_predictions - flat_targets) ** 2
    metrics = {}
    if torch.any(boundary):
        metrics['loss_boundary'] = squared_error[boundary].mean().item()
    if torch.any(interior):
        metrics['loss_interior'] = squared_error[interior].mean().item()
    return metrics


def compute_channel_losses(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute loss for each channel separately.
    
    Args:
        predictions: Model predictions [batch, channels, height, width]
        targets: Ground truth [batch, channels, height, width]
    
    Returns:
        Dictionary with channel losses
    """
    channel_losses = {}
    
    if predictions.dim() in (4, 5) and targets.dim() == predictions.dim():
        valid_mask = _validated_mask(targets, mask)
        channel_axis = 1 if predictions.dim() == 4 else 2
        n_channels = predictions.shape[channel_axis]

        for i in range(n_channels):
            if predictions.dim() == 4:
                channel_predictions = predictions[:, i]
                channel_targets = targets[:, i]
                channel_mask = valid_mask[:, i]
            else:
                channel_predictions = predictions[:, :, i]
                channel_targets = targets[:, :, i]
                channel_mask = valid_mask[:, :, i]
            if not torch.any(channel_mask):
                continue
            channel_loss = F.mse_loss(
                channel_predictions[channel_mask], channel_targets[channel_mask]
            )
            channel_losses[f'loss_channel_{i}'] = channel_loss.item()
    
    return channel_losses


def compute_quantile_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                           quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
                           mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute quantile metrics of prediction errors.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        quantiles: List of quantiles to compute
    
    Returns:
        Dictionary with quantile metrics
    """
    valid_mask = _validated_mask(targets, mask)
    errors = torch.abs(predictions - targets)[valid_mask]
    if errors.numel() == 0:
        raise ValueError("Cannot calculate quantiles: no valid cells")
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

def get_loss(loss_type, model=None):
    """Build a loss function from a config entry.

    ``loss_type`` is either a plain string (``"mse"``/``"mae"``) for the
    original pointwise losses, or a mapping selecting the residual-aware loss:

        loss:
          type: mse
          vertical_weight: 1.0
          tail_weight: 0.5
          tail_fraction: 0.05

    When ``model`` is supplied its per-layer pressure sigmas are handed to the
    loss so the vertical term is computed in physical rather than scaled units.
    """
    if isinstance(loss_type, dict):
        config = dict(loss_type)
        base = config.pop("type", "mse")
        if config.pop("layer_sigmas_from_model", True) and model is not None:
            config["layer_sigmas"] = pressure_layer_sigmas(model)
        else:
            config.pop("layer_sigmas", None)
        # The ponding hinge needs the scaled location of physical p=0 in the
        # top layer; read it from the same scalers the model applies.
        if (
            config.pop("top_zero_scaled_from_model", True)
            and model is not None
            and config.get("ponding_weight", 0.0) > 0.0
        ):
            config.setdefault(
                "top_zero_scaled", pressure_top_zero_scaled(model)
            )
        return PressureFirstGuessLoss(base=base, **config)

    if loss_type == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss_type == "mae":
        loss_fn = torch.nn.L1Loss()
    else:
        raise ValueError(f"Loss {loss_type} not supported")
    return loss_fn


def pressure_layer_sigmas(model):
    """Per-layer pressure standard deviations from a model's scalers.

    Uses the key mapping the model resolved at construction so this agrees with
    ``scale_pressure`` even when a scaler set still uses the legacy
    ``press_diff_*`` names.
    """
    names = getattr(model, "pressure_scaler_names", None)
    if not names:
        names = resolve_pressure_scaler_names(model.scalers)
    if not names:
        raise ValueError("Model scalers contain no pressure_* entries")
    return [float(model.scalers[name][1]) for name in names]


def pressure_top_zero_scaled(model):
    """Scaled-space location of physical zero pressure in the top layer.

    The top layer's sign selects the overland-flow branch of the Richards
    residual, so the ponding hinge and flip metric need to know where p=0
    lands after scaling: ``(0 - mu) / sigma`` of the top layer's scaler.
    """
    names = getattr(model, "pressure_scaler_names", None)
    if not names:
        names = resolve_pressure_scaler_names(model.scalers)
    if not names:
        raise ValueError("Model scalers contain no pressure_* entries")
    mu, sigma = (float(v) for v in model.scalers[names[-1]][:2])
    if sigma <= 0:
        raise ValueError("Top pressure layer sigma must be positive")
    return -mu / sigma


def ponding_flip_sums(
    predictions, targets, persistence, mask, top_zero_scaled
):
    """Top-layer ponding-state disagreements with the target.

    A "flip" is a valid top-layer cell whose predicted physical pressure sign
    differs from the target's.  Persistence flips are counted from the same
    cells as the reference: the benchmarked model produced ~90 flips per field
    per hour against persistence's ~2-6, and every flip puts the solver on the
    wrong overland-flow branch.  A checkpoint whose flip count is not at or
    below persistence's is not worth benchmarking.

    Args:
        predictions: Model output, ``(batch, z, y, x)``, scaled units.
        targets: True next state, same shape.
        persistence: The previous state, i.e. what the solver would have used.
        mask: Bool tensor of valid cells, same shape.
        top_zero_scaled: Scaled-space value of physical zero pressure in the
            top layer.

    Returns:
        ``(model_flips, persistence_flips, valid_cells)`` as floats.
    """
    top_mask = mask[:, -1].to(torch.bool)
    if not torch.any(top_mask):
        return 0.0, 0.0, 0.0
    target_ponded = targets[:, -1] > top_zero_scaled
    model_flips = (
        ((predictions[:, -1] > top_zero_scaled) != target_ponded)[top_mask]
        .sum()
        .item()
    )
    persistence_flips = (
        ((persistence[:, -1] > top_zero_scaled) != target_ponded)[top_mask]
        .sum()
        .item()
    )
    return float(model_flips), float(persistence_flips), float(top_mask.sum().item())
