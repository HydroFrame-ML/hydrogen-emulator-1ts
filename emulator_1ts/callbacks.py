import os
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from logger import info, verbose, error
import copy


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Dict[str, Any] = None):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the end of each batch."""
        pass


class CallbackManager:
    """Manages multiple callbacks."""
    
    def __init__(self, callbacks: List[Callback] = None):
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: Callback):
        """Add a callback to the manager."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class EarlyStoppingCallback(Callback):
    """Stop training when a monitored metric has stopped improving."""
    
    def __init__(self, 
                 monitor: str = 'val_loss',
                 patience: int = 10,
                 min_delta: float = 1e-6,
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max' - whether to minimize or maximize the monitored metric
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.model = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        else:
            self.monitor_op = np.greater
            self.best = -np.inf
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        self.wait = 0
        self.stopped_epoch = 0
        self.model = logs.get('model') if logs else None
        
        if self.mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if not logs:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            verbose(f"Early stopping conditioned on metric `{self.monitor}` which is not available. Skipping.")
            return
        
        if self.monitor_op(float(current) - float(self.min_delta), self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights and self.model:
                self.best_weights = copy.deepcopy(self.model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logs['stop_training'] = True
                info(f"Early stopping triggered at epoch {epoch + 1}")
                
                if self.restore_best_weights and self.best_weights and self.model:
                    info("Restoring model weights from the best epoch")
                    self.model.load_state_dict(self.best_weights)


class ModelCheckpointCallback(Callback):
    """Save the model after every epoch."""
    
    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 save_best_only: bool = True,
                 save_weights_only: bool = False,
                 mode: str = 'min',
                 save_top_k: int = 1,
                 verbose_mode: bool = True):
        """
        Args:
            filepath: Path to save the model file
            monitor: Metric to monitor
            save_best_only: Only save when the model is considered the "best"
            save_weights_only: Save only the model weights
            mode: 'min' or 'max'
            save_top_k: Number of best models to keep
            verbose_mode: Whether to print save messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.save_top_k = save_top_k
        self.verbose_mode = verbose_mode
        
        self.model = None
        self.saved_models = []  # List of (score, filepath) tuples
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        else:
            self.monitor_op = np.greater
            self.best = -np.inf
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        self.model = logs.get('model') if logs else None
        if self.mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if not logs or not self.model:
            return
        
        current = logs.get(self.monitor)
        if current is None and self.save_best_only:
            verbose(f"Model checkpoint conditioned on metric `{self.monitor}` which is not available. Skipping.")
            return
        
        # Generate filepath with epoch number
        filepath = self.filepath.replace('.pth', f'_epoch_{epoch:03d}.pth')
        
        if self.save_best_only:
            if current is None:
                return
            
            if self.monitor_op(current, self.best):
                self.best = current
                self._save_model(filepath, current, epoch)
        else:
            self._save_model(filepath, current, epoch)
    
    def _save_model(self, filepath: str, score: float, epoch: int):
        """Save the model and manage top-k models."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.save_weights_only:
            torch.save(self.model.state_dict(), filepath)
        else:
            # Save TorchScript model
            model_copy = copy.deepcopy(self.model).to(torch.float64)
            scripted_model = torch.jit.script(model_copy)
            torch.jit.save(scripted_model, filepath)
        
        if self.verbose_mode:
            verbose(f"Model checkpoint saved to {filepath} (epoch {epoch + 1}, {self.monitor}: {score:.6f})")
        
        # Manage top-k models
        self.saved_models.append((score, filepath))
        
        # Sort by score (best first)
        if self.mode == 'min':
            self.saved_models.sort(key=lambda x: x[0])
        else:
            self.saved_models.sort(key=lambda x: x[0], reverse=True)
        
        # Remove excess models
        while len(self.saved_models) > self.save_top_k:
            _, old_filepath = self.saved_models.pop()
            if os.path.exists(old_filepath):
                os.remove(old_filepath)
                if self.verbose_mode:
                    verbose(f"Removed old checkpoint: {old_filepath}")


class LearningRateSchedulerCallback(Callback):
    """Learning rate scheduler callback."""
    
    def __init__(self, scheduler, monitor: str = None):
        """
        Args:
            scheduler: PyTorch learning rate scheduler
            monitor: Metric to monitor (for ReduceLROnPlateau)
        """
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if hasattr(self.scheduler, 'step'):
            if self.monitor and logs:
                # For ReduceLROnPlateau
                metric_value = logs.get(self.monitor)
                if metric_value is not None:
                    self.scheduler.step(metric_value)
                else:
                    verbose(f"LR scheduler metric `{self.monitor}` not found in logs")
            else:
                # For other schedulers
                self.scheduler.step()
        
        # Log current learning rate
        if logs and hasattr(self.scheduler, 'get_last_lr'):
            current_lr = self.scheduler.get_last_lr()[0]
            logs['learning_rate'] = current_lr


class GradientClippingCallback(Callback):
    """Gradient clipping callback."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Args:
            max_norm: Maximum norm of the gradients
            norm_type: Type of the used p-norm
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.model = None
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        self.model = logs.get('model') if logs else None
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        if self.model and logs and logs.get('training', False):
            # Clip gradients after backward pass
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.max_norm, 
                norm_type=self.norm_type
            )
            if logs:
                logs['grad_norm'] = grad_norm.item()


def create_callbacks_from_config(config: Dict[str, Any], model, log_location: str, name: str) -> List[Callback]:
    """Create callbacks from configuration."""
    callbacks = []
    callback_config = config.get('callbacks', {})
    
    # Early Stopping
    if callback_config.get('early_stopping', {}).get('enabled', False):
        es_config = callback_config['early_stopping']
        callbacks.append(EarlyStoppingCallback(
            monitor=es_config.get('monitor', 'val_loss'),
            patience=es_config.get('patience', 10),
            min_delta=es_config.get('min_delta', 1e-6),
            mode=es_config.get('mode', 'min'),
            restore_best_weights=es_config.get('restore_best_weights', True)
        ))
        info(f"Early stopping enabled: monitor={es_config.get('monitor', 'val_loss')}, patience={es_config.get('patience', 10)}")
    
    # Model Checkpoint
    if callback_config.get('model_checkpoint', {}).get('enabled', False):
        mc_config = callback_config['model_checkpoint']
        checkpoint_path = os.path.join(log_location, f"{name}_checkpoint.pth")
        callbacks.append(ModelCheckpointCallback(
            filepath=checkpoint_path,
            monitor=mc_config.get('monitor', 'val_loss'),
            save_best_only=mc_config.get('save_best_only', True),
            save_weights_only=mc_config.get('save_weights_only', False),
            mode=mc_config.get('mode', 'min'),
            save_top_k=mc_config.get('save_top_k', 1),
            verbose_mode=mc_config.get('verbose', True)
        ))
        info(f"Model checkpointing enabled: monitor={mc_config.get('monitor', 'val_loss')}, save_best_only={mc_config.get('save_best_only', True)}")
    
    # Gradient Clipping
    if callback_config.get('gradient_clipping', {}).get('enabled', False):
        gc_config = callback_config['gradient_clipping']
        callbacks.append(GradientClippingCallback(
            max_norm=gc_config.get('max_norm', 1.0),
            norm_type=gc_config.get('norm_type', 2.0)
        ))
        info(f"Gradient clipping enabled: max_norm={gc_config.get('max_norm', 1.0)}")
    
    return callbacks
