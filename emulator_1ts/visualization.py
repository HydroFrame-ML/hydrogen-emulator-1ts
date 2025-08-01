import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.figure import Figure
import seaborn as sns
from typing import Tuple, Optional, List
import io
from PIL import Image


def set_style():
    """Set consistent plotting style."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })


def fig_to_image(fig: Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img


def create_timeseries_plot(predictions: torch.Tensor, 
                          targets: torch.Tensor, 
                          sample_indices: List[int] = None,
                          channel_idx: int = 0,
                          max_samples: int = 3) -> Figure:
    """
    Create timeseries comparison plot for selected samples.
    
    Args:
        predictions: Model predictions [batch, channels, height, width]
        targets: Ground truth [batch, channels, height, width]
        sample_indices: Which samples to plot (if None, select randomly)
        channel_idx: Which channel/depth layer to plot
        max_samples: Maximum number of samples to plot
    
    Returns:
        matplotlib Figure
    """
    set_style()
    
    if sample_indices is None:
        n_samples = min(max_samples, predictions.shape[0])
        sample_indices = np.random.choice(predictions.shape[0], n_samples, replace=False)
    
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12, 3 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = [axes]
    
    for i, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
        # Extract spatial mean for the timeseries (treating spatial dims as time)
        pred_sample = predictions[sample_idx, channel_idx].cpu().numpy()
        target_sample = targets[sample_idx, channel_idx].cpu().numpy()
        
        # Flatten spatial dimensions and treat as time series
        pred_ts = pred_sample.flatten()
        target_ts = target_sample.flatten()
        
        time_steps = np.arange(len(pred_ts))
        
        ax.plot(time_steps, target_ts, label='Ground Truth', alpha=0.7, linewidth=1.5)
        ax.plot(time_steps, pred_ts, label='Prediction', alpha=0.7, linewidth=1.5)
        
        ax.set_title(f'Sample {sample_idx}, Channel {channel_idx}')
        ax.set_xlabel('Spatial Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_spatial_field_plot(predictions: torch.Tensor, 
                             targets: torch.Tensor,
                             channel_idx: int = 0,
                             sample_idx: int = 0,
                             vmin: float = None,
                             vmax: float = None) -> Figure:
    """
    Create spatial field comparison plot.
    
    Args:
        predictions: Model predictions [batch, channels, height, width]
        targets: Ground truth [batch, channels, height, width]
        channel_idx: Which channel/depth layer to plot
        sample_idx: Which sample to plot
        vmin, vmax: Color scale limits
    
    Returns:
        matplotlib Figure
    """
    set_style()
    
    pred_field = predictions[sample_idx, channel_idx].cpu().numpy()
    target_field = targets[sample_idx, channel_idx].cpu().numpy()
    diff_field = pred_field - target_field
    
    if vmin is None:
        vmin = min(pred_field.min(), target_field.min())
    if vmax is None:
        vmax = max(pred_field.max(), target_field.max())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Ground truth
    im1 = axes[0].imshow(target_field, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # Prediction
    im2 = axes[1].imshow(pred_field, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Prediction')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    diff_max = max(abs(diff_field.min()), abs(diff_field.max()))
    im3 = axes[2].imshow(diff_field, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    axes[2].set_title('Difference (Pred - Truth)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f'Spatial Fields - Sample {sample_idx}, Channel {channel_idx}')
    plt.tight_layout()
    return fig


def create_error_distribution_plot(predictions: torch.Tensor, 
                                  targets: torch.Tensor,
                                  channel_idx: int = None) -> Figure:
    """
    Create error distribution histogram.
    
    Args:
        predictions: Model predictions [batch, channels, height, width]
        targets: Ground truth [batch, channels, height, width]
        channel_idx: Which channel to plot (if None, plot all channels)
    
    Returns:
        matplotlib Figure
    """
    set_style()
    
    errors = (predictions - targets).cpu().numpy()
    
    if channel_idx is not None:
        errors = errors[:, channel_idx]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        errors_flat = errors.flatten()
        ax.hist(errors_flat, bins=50, alpha=0.7, density=True, edgecolor='black')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Density')
        ax.set_title(f'Error Distribution - Channel {channel_idx}')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(errors_flat)
        std_error = np.std(errors_flat)
        ax.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.4f}')
        ax.axvline(mean_error + std_error, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_error:.4f}')
        ax.axvline(mean_error - std_error, color='orange', linestyle='--', alpha=0.7)
        ax.legend()
        
    else:
        # Plot all channels
        n_channels = errors.shape[1]
        n_cols = min(4, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_channels):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            channel_errors = errors[:, i].flatten()
            ax.hist(channel_errors, bins=30, alpha=0.7, density=True, edgecolor='black')
            ax.set_xlabel('Error')
            ax.set_ylabel('Density')
            ax.set_title(f'Channel {i}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_channels, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('Error Distributions by Channel')
    
    plt.tight_layout()
    return fig


def create_channel_performance_heatmap(channel_losses: dict, 
                                      quantile_metrics: dict = None) -> Figure:
    """
    Create heatmap showing performance by channel.
    
    Args:
        channel_losses: Dictionary with channel losses
        quantile_metrics: Dictionary with quantile metrics
    
    Returns:
        matplotlib Figure
    """
    set_style()
    
    # Extract channel losses
    channels = []
    losses = []
    for key, value in channel_losses.items():
        if key.startswith('loss_channel_'):
            channel_num = int(key.split('_')[-1])
            channels.append(channel_num)
            losses.append(value)
    
    if not channels:
        # Create empty plot if no channel data
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No channel loss data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Channel Performance')
        return fig
    
    # Sort by channel number
    sorted_data = sorted(zip(channels, losses))
    channels, losses = zip(*sorted_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Channel losses bar plot
    axes[0].bar(range(len(channels)), losses, alpha=0.7)
    axes[0].set_xlabel('Channel (Depth Layer)')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss by Channel')
    axes[0].set_xticks(range(len(channels)))
    axes[0].set_xticklabels([f'Ch {c}' for c in channels], rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Quantile metrics if available
    if quantile_metrics:
        quantiles = []
        values = []
        for key, value in quantile_metrics.items():
            if key.startswith('error_q'):
                q_num = key.split('q')[-1]
                quantiles.append(f'Q{q_num}')
                values.append(value)
        
        if quantiles:
            axes[1].bar(range(len(quantiles)), values, alpha=0.7, color='orange')
            axes[1].set_xlabel('Quantile')
            axes[1].set_ylabel('Error Value')
            axes[1].set_title('Error Quantiles')
            axes[1].set_xticks(range(len(quantiles)))
            axes[1].set_xticklabels(quantiles)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No quantile data available', 
                        ha='center', va='center', transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, 'No quantile data available', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    return fig


def create_training_progress_plot(train_losses: List[float], 
                                 val_losses: List[float] = None,
                                 learning_rates: List[float] = None) -> Figure:
    """
    Create training progress plot.
    
    Args:
        train_losses: Training losses by epoch
        val_losses: Validation losses by epoch
        learning_rates: Learning rates by epoch
    
    Returns:
        matplotlib Figure
    """
    set_style()
    
    n_plots = 1 + (val_losses is not None) + (learning_rates is not None)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    epochs = range(1, len(train_losses) + 1)
    
    # Training loss
    axes[plot_idx].plot(epochs, train_losses, label='Training Loss', linewidth=2)
    if val_losses:
        axes[plot_idx].plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    axes[plot_idx].set_xlabel('Epoch')
    axes[plot_idx].set_ylabel('Loss')
    axes[plot_idx].set_title('Training Progress')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].set_yscale('log')
    plot_idx += 1
    
    # Learning rate
    if learning_rates:
        axes[plot_idx].plot(epochs, learning_rates, color='green', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Learning Rate')
        axes[plot_idx].set_title('Learning Rate Schedule')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_yscale('log')
    
    plt.tight_layout()
    return fig
