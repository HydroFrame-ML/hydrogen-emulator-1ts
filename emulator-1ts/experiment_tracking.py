import os
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from logger import info, verbose, error
from visualization import (
    create_timeseries_plot, 
    create_spatial_field_plot, 
    create_error_distribution_plot,
    create_channel_performance_heatmap,
    create_training_progress_plot,
    fig_to_image
)
from callbacks import Callback


class TensorBoardTracker(Callback):
    """TensorBoard experiment tracking callback."""
    
    def __init__(self, 
                 log_dir: str,
                 experiment_name: str,
                 config: Dict[str, Any] = None,
                 log_hyperparams: bool = True,
                 log_model_graph: bool = True,
                 log_images: bool = True,
                 log_frequency: int = 10,
                 max_images: int = 5):
        """
        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name of the experiment
            config: Configuration dictionary to log as hyperparameters
            log_hyperparams: Whether to log hyperparameters
            log_model_graph: Whether to log model architecture
            log_images: Whether to log visualization images
            log_frequency: How often to log images (every N epochs)
            max_images: Maximum number of images to log per visualization type
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.config = config or {}
        self.log_hyperparams = log_hyperparams
        self.log_model_graph = log_model_graph
        self.log_images = log_images
        self.log_frequency = log_frequency
        self.max_images = max_images
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        
        self.writer = None
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Sample data for visualizations
        self.sample_predictions = None
        self.sample_targets = None
        
    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Initialize TensorBoard writer and log initial information."""
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.experiment_dir)
        
        if logs:
            self.model = logs.get('model')
        
        info(f"TensorBoard logging initialized: {self.experiment_dir}")
        
        # Log hyperparameters
        if self.log_hyperparams and self.config:
            self._log_hyperparameters()
        
        # Log model graph
        if self.log_model_graph and self.model:
            self._log_model_graph()
    
    def on_train_end(self, logs: Dict[str, Any] = None):
        """Close TensorBoard writer and create final visualizations."""
        if self.writer:
            # Create final training progress plot
            if self.train_losses:
                self._log_training_progress()
            
            self.writer.close()
            info(f"TensorBoard logging completed. View with: tensorboard --logdir {self.experiment_dir}")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Log metrics and visualizations at the end of each epoch."""
        if not self.writer or not logs:
            return
        
        # Log scalar metrics
        self._log_scalar_metrics(epoch, logs)
        
        # Store losses for progress plot
        if 'train_loss' in logs:
            self.train_losses.append(logs['train_loss'])
        if 'val_loss' in logs:
            self.val_losses.append(logs['val_loss'])
        if 'learning_rate' in logs:
            self.learning_rates.append(logs['learning_rate'])
        
        # Log images periodically
        if self.log_images and (epoch + 1) % self.log_frequency == 0:
            self._log_visualizations(epoch, logs)
    
    def _log_hyperparameters(self):
        """Log hyperparameters to TensorBoard."""
        try:
            # Flatten nested config and convert to appropriate types
            hparams = self._flatten_config(self.config)
            
            # Add derived hyperparameters
            derived_hparams = self._compute_derived_hyperparameters(self.config)
            hparams.update(derived_hparams)
            
            # Convert values to appropriate types for TensorBoard
            hparams_clean = {}
            for key, value in hparams.items():
                if isinstance(value, (int, float, str, bool)):
                    hparams_clean[key] = value
                elif isinstance(value, (list, tuple)):
                    # Convert lists/tuples to string representation
                    if len(str(value)) < 100:  # Only if not too long
                        hparams_clean[key] = str(value)
                    else:
                        hparams_clean[f"{key}_length"] = len(value)
                elif isinstance(value, dict):
                    # For nested dicts, add summary info
                    hparams_clean[f"{key}_keys"] = str(list(value.keys())[:5])  # First 5 keys
                else:
                    hparams_clean[key] = str(value)
            
            # Log as both hparams and text summary
            self.writer.add_hparams(hparams_clean, {})
            
            # Also log as text for better readability
            hparams_text = self._format_hyperparameters_text(hparams_clean)
            self.writer.add_text("Hyperparameters/Config", hparams_text, 0)
            
            verbose("Hyperparameters logged to TensorBoard")
            
        except Exception as e:
            error(f"Failed to log hyperparameters: {e}")
    
    def _log_model_graph(self):
        """Log model architecture to TensorBoard."""
        try:
            if self.model:
                # Create dummy input to trace the model
                # Assuming the model expects (state, evaptrans, params)
                dummy_state = torch.randn(1, 10, 32, 32)  # Adjust dimensions as needed
                dummy_evaptrans = torch.randn(1, 4, 32, 32)  # Adjust based on n_evaptrans
                dummy_params = torch.randn(1, 37, 32, 32)  # Adjust based on parameter count
                
                # Move to same device as model
                device = next(self.model.parameters()).device
                dummy_state = dummy_state.to(device)
                dummy_evaptrans = dummy_evaptrans.to(device)
                dummy_params = dummy_params.to(device)
                
                self.writer.add_graph(self.model, (dummy_state, dummy_evaptrans, dummy_params))
                verbose("Model graph logged to TensorBoard")
                
        except Exception as e:
            error(f"Failed to log model graph: {e}")
    
    def _log_scalar_metrics(self, epoch: int, logs: Dict[str, Any]):
        """Log scalar metrics to TensorBoard."""
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not key.startswith('_'):
                # Organize metrics into categories
                if 'loss' in key and not key.startswith('loss_channel_'):
                    self.writer.add_scalar(f'Loss/{key}', value, epoch)
                elif 'error_q' in key:
                    self.writer.add_scalar(f'Quantiles/{key}', value, epoch)
                elif 'loss_channel_' in key:
                    channel_num = key.split('_')[-1]
                    self.writer.add_scalar(f'Channel_Loss/channel_{channel_num}', value, epoch)
                elif key == 'learning_rate':
                    self.writer.add_scalar('Training/learning_rate', value, epoch)
                elif key == 'grad_norm':
                    self.writer.add_scalar('Training/gradient_norm', value, epoch)
                elif key in ['weight_decay', 'momentum', 'beta1', 'beta2', 'optimizer_eps']:
                    self.writer.add_scalar(f'Optimizer/{key}', value, epoch)
                elif key.startswith('scheduler_'):
                    self.writer.add_scalar(f'Scheduler/{key}', value, epoch)
                elif key == 'scheduler_type':
                    # Skip string values for scalar logging
                    continue
                else:
                    self.writer.add_scalar(f'Metrics/{key}', value, epoch)
    
    def _log_visualizations(self, epoch: int, logs: Dict[str, Any]):
        """Log visualization images to TensorBoard."""
        try:
            # Get sample predictions and targets if available
            predictions = logs.get('sample_predictions')
            targets = logs.get('sample_targets')
            
            if predictions is not None and targets is not None:
                self._log_prediction_visualizations(epoch, predictions, targets, logs)
            
            # Log channel performance heatmap
            channel_losses = {k: v for k, v in logs.items() if k.startswith('loss_channel_')}
            quantile_metrics = {k: v for k, v in logs.items() if k.startswith('error_q')}
            
            if channel_losses:
                self._log_channel_performance(epoch, channel_losses, quantile_metrics)
                
        except Exception as e:
            error(f"Failed to log visualizations: {e}")
    
    def _log_prediction_visualizations(self, epoch: int, predictions: torch.Tensor, 
                                     targets: torch.Tensor, logs: Dict[str, Any]):
        """Log prediction visualization images."""
        try:
            # Ensure tensors are on CPU
            predictions = predictions.cpu()
            targets = targets.cpu()
            
            # Limit number of samples to avoid memory issues
            max_samples = min(self.max_images, predictions.shape[0])
            
            # Timeseries plots for different channels
            for channel_idx in [0, predictions.shape[1]//2, predictions.shape[1]-1]:
                if channel_idx < predictions.shape[1]:
                    fig = create_timeseries_plot(
                        predictions[:max_samples], 
                        targets[:max_samples], 
                        channel_idx=channel_idx,
                        max_samples=3
                    )
                    self.writer.add_figure(f'Timeseries/channel_{channel_idx}', fig, epoch)
                    plt.close(fig)
            
            # Spatial field plots
            for channel_idx in [0, predictions.shape[1]//2, predictions.shape[1]-1]:
                if channel_idx < predictions.shape[1]:
                    fig = create_spatial_field_plot(
                        predictions, 
                        targets, 
                        channel_idx=channel_idx,
                        sample_idx=0
                    )
                    self.writer.add_figure(f'Spatial_Fields/channel_{channel_idx}', fig, epoch)
                    plt.close(fig)
            
            # Error distribution plots
            fig = create_error_distribution_plot(predictions, targets)
            self.writer.add_figure('Error_Distributions/all_channels', fig, epoch)
            plt.close(fig)
            
            verbose(f"Prediction visualizations logged for epoch {epoch}")
            
        except Exception as e:
            error(f"Failed to log prediction visualizations: {e}")
    
    def _log_channel_performance(self, epoch: int, channel_losses: Dict[str, float], 
                                quantile_metrics: Dict[str, float]):
        """Log channel performance heatmap."""
        try:
            fig = create_channel_performance_heatmap(channel_losses, quantile_metrics)
            self.writer.add_figure('Performance/channel_performance', fig, epoch)
            plt.close(fig)
            verbose(f"Channel performance logged for epoch {epoch}")
            
        except Exception as e:
            error(f"Failed to log channel performance: {e}")
    
    def _log_training_progress(self):
        """Log final training progress plot."""
        try:
            val_losses = self.val_losses if self.val_losses else None
            learning_rates = self.learning_rates if self.learning_rates else None
            
            fig = create_training_progress_plot(
                self.train_losses, 
                val_losses, 
                learning_rates
            )
            self.writer.add_figure('Training/progress', fig, len(self.train_losses))
            plt.close(fig)
            verbose("Training progress plot logged")
            
        except Exception as e:
            error(f"Failed to log training progress: {e}")
    
    def _flatten_config(self, config: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested configuration dictionary."""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _compute_derived_hyperparameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Compute derived hyperparameters from the configuration."""
        derived = {}
        
        try:
            # Model complexity metrics
            model_def = config.get('model_def', {})
            if model_def:
                # Compute approximate model parameters
                in_channels = model_def.get('in_channels', 0)
                out_channels = model_def.get('out_channels', 0)
                hidden_dim = model_def.get('hidden_dim', 0)
                depth = model_def.get('depth', 0)
                kernel_size = model_def.get('kernel_size', 3)
                
                # Rough parameter count estimation for ResNet-like architecture
                if in_channels and hidden_dim and depth:
                    # First conv layer
                    first_conv_params = in_channels * hidden_dim * kernel_size * kernel_size
                    # ResNet blocks (rough estimation)
                    resnet_params = depth * (hidden_dim * hidden_dim * kernel_size * kernel_size * 2)
                    # Final conv layer
                    final_conv_params = hidden_dim * out_channels * kernel_size * kernel_size
                    
                    total_params = first_conv_params + resnet_params + final_conv_params
                    derived['estimated_model_params'] = total_params
                    derived['model_complexity_score'] = depth * hidden_dim
                
                # Model architecture ratios
                if in_channels and out_channels:
                    derived['channel_compression_ratio'] = in_channels / out_channels
                if hidden_dim and in_channels:
                    derived['hidden_expansion_ratio'] = hidden_dim / in_channels
            
            # Data configuration metrics
            data_def = config.get('data_def', {})
            if data_def:
                patch_size_x = data_def.get('patch_size_x', 0)
                patch_size_y = data_def.get('patch_size_y', 0)
                n_evaptrans = data_def.get('n_evaptrans', 0)
                
                if patch_size_x and patch_size_y:
                    derived['patch_area'] = patch_size_x * patch_size_y
                    derived['patch_aspect_ratio'] = patch_size_x / patch_size_y
                
                parameter_list = data_def.get('parameter_list', [])
                if parameter_list:
                    derived['n_parameters'] = len(parameter_list)
                
                if n_evaptrans:
                    derived['evaptrans_channels'] = n_evaptrans
            
            # Training configuration metrics
            batch_size = config.get('batch_size', 0)
            lr = config.get('lr', 0)
            n_epochs = config.get('n_epochs', 0)
            
            if batch_size and lr:
                derived['lr_batch_ratio'] = lr * batch_size  # Effective learning rate
            
            if n_epochs:
                derived['total_epochs'] = n_epochs
            
            # Callback configuration summary
            callbacks_config = config.get('callbacks', {})
            if callbacks_config:
                enabled_callbacks = []
                for callback_name, callback_config in callbacks_config.items():
                    if isinstance(callback_config, dict) and callback_config.get('enabled', False):
                        enabled_callbacks.append(callback_name)
                derived['enabled_callbacks_count'] = len(enabled_callbacks)
                
                # Early stopping configuration
                if callbacks_config.get('early_stopping', {}).get('enabled', False):
                    es_config = callbacks_config['early_stopping']
                    derived['early_stopping_patience'] = es_config.get('patience', 0)
                
                # Learning rate scheduler configuration
                if callbacks_config.get('lr_scheduler', {}).get('enabled', False):
                    lr_config = callbacks_config['lr_scheduler']
                    derived['lr_scheduler_type'] = lr_config.get('type', 'unknown')
                    derived['lr_scheduler_patience'] = lr_config.get('patience', 0)
                    derived['lr_scheduler_factor'] = lr_config.get('factor', 1.0)
            
            # TensorBoard configuration
            tensorboard_config = config.get('tensorboard', {})
            if tensorboard_config:
                derived['tensorboard_log_frequency'] = tensorboard_config.get('log_frequency', 10)
                derived['tensorboard_max_images'] = tensorboard_config.get('max_images', 5)
            
        except Exception as e:
            error(f"Failed to compute derived hyperparameters: {e}")
        
        return derived
    
    def _format_hyperparameters_text(self, hparams: Dict[str, Any]) -> str:
        """Format hyperparameters as readable text."""
        lines = ["# Experiment Configuration\n"]
        
        # Group parameters by category
        categories = {
            'Model': [],
            'Training': [],
            'Data': [],
            'Callbacks': [],
            'TensorBoard': [],
            'Derived': [],
            'Other': []
        }
        
        for key, value in sorted(hparams.items()):
            if any(x in key.lower() for x in ['model', 'channel', 'hidden', 'depth', 'kernel']):
                categories['Model'].append(f"- **{key}**: {value}")
            elif any(x in key.lower() for x in ['lr', 'batch', 'epoch', 'optimizer', 'loss']):
                categories['Training'].append(f"- **{key}**: {value}")
            elif any(x in key.lower() for x in ['data', 'patch', 'parameter', 'evaptrans']):
                categories['Data'].append(f"- **{key}**: {value}")
            elif any(x in key.lower() for x in ['callback', 'early', 'scheduler', 'gradient']):
                categories['Callbacks'].append(f"- **{key}**: {value}")
            elif 'tensorboard' in key.lower():
                categories['TensorBoard'].append(f"- **{key}**: {value}")
            elif any(x in key.lower() for x in ['estimated', 'complexity', 'ratio', 'enabled_callbacks']):
                categories['Derived'].append(f"- **{key}**: {value}")
            else:
                categories['Other'].append(f"- **{key}**: {value}")
        
        # Format each category
        for category, items in categories.items():
            if items:
                lines.append(f"\n## {category}\n")
                lines.extend(items)
        
        return '\n'.join(lines)


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
            channel_loss = torch.nn.functional.mse_loss(
                predictions[:, i], 
                targets[:, i]
            )
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
        except Exception as e:
            error(f"Failed to compute quantile {q}: {e}")
    
    return quantile_metrics


def create_tensorboard_tracker_from_config(config: Dict[str, Any], 
                                          experiment_name: str) -> Optional[TensorBoardTracker]:
    """
    Create TensorBoard tracker from configuration.
    
    Args:
        config: Configuration dictionary
        experiment_name: Name of the experiment
    
    Returns:
        TensorBoardTracker instance or None if disabled
    """
    tensorboard_config = config.get('tensorboard', {})
    
    if not tensorboard_config.get('enabled', True):
        return None
    
    log_dir = config.get('tensorboard_log_dir', './runs')
    
    tracker = TensorBoardTracker(
        log_dir=log_dir,
        experiment_name=experiment_name,
        config=config,
        log_hyperparams=tensorboard_config.get('log_hyperparams', True),
        log_model_graph=tensorboard_config.get('log_model_graph', True),
        log_images=tensorboard_config.get('log_images', True),
        log_frequency=tensorboard_config.get('log_frequency', 10),
        max_images=tensorboard_config.get('max_images', 5)
    )
    
    info(f"TensorBoard tracker created: {tracker.experiment_dir}")
    return tracker
