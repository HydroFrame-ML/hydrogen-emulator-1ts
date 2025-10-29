"""
Progressive timestep training manager for curriculum learning in multistep models.
"""
import torch
from typing import List, Optional, Dict, Any
from .logger import info, verbose, error


class ProgressiveTimestepManager:
    """
    Manages progressive timestep rollout training where the model trains on 
    increasing sequence lengths as training progresses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the progressive timestep manager.
        
        Args:
            config: Configuration dictionary containing progressive_timesteps settings
        """
        self.config = config
        self.progressive_config = config.get('progressive_timesteps', {})
        self.enabled = self.progressive_config.get('enabled', False)
        
        if self.enabled:
            self.timesteps = self.progressive_config.get('timesteps', [])
            self.switch_epochs = self.progressive_config.get('switch_epochs', [])
            
            # Validate configuration
            self._validate_config()
            
            self.current_timestep_index = 0
            self.current_timesteps = self.timesteps[0] if self.timesteps else 1
            
            info(f"Progressive timestep training enabled")
            info(f"Timesteps: {self.timesteps}")
            info(f"Switch epochs: {self.switch_epochs}")
        else:
            # Fallback to traditional single timestep configuration
            self.timesteps = [config.get('data_def', {}).get('n_timesteps', 1)]
            self.switch_epochs = [0]
            self.current_timestep_index = 0
            self.current_timesteps = self.timesteps[0]
            
            verbose("Progressive timestep training disabled, using fixed timesteps")
    
    def _validate_config(self):
        """Validate the progressive timestep configuration."""
        if not isinstance(self.timesteps, list) or len(self.timesteps) == 0:
            raise ValueError("progressive_timesteps.timesteps must be a non-empty list")
        
        if not isinstance(self.switch_epochs, list) or len(self.switch_epochs) == 0:
            raise ValueError("progressive_timesteps.switch_epochs must be a non-empty list")
        
        if len(self.timesteps) != len(self.switch_epochs):
            raise ValueError("Length of timesteps and switch_epochs must match")
        
        # Ensure timesteps are positive
        if any(t <= 0 for t in self.timesteps):
            raise ValueError("All timesteps must be positive integers")
        
        # Ensure switch epochs are non-negative and increasing
        if any(e < 0 for e in self.switch_epochs):
            raise ValueError("All switch epochs must be non-negative")
        
        if self.switch_epochs != sorted(self.switch_epochs):
            raise ValueError("Switch epochs must be in ascending order")
        
        # Warn if timesteps are not increasing (though not strictly required)
        if self.timesteps != sorted(self.timesteps):
            verbose("WARNING: Timesteps are not in ascending order - this may not follow curriculum learning principles")
    
    def should_switch_timesteps(self, current_epoch: int) -> bool:
        """
        Check if we should switch to the next timestep length.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
            
        Returns:
            True if we should switch timesteps
        """
        if not self.enabled:
            return False
        
        # Check if we've reached a switch epoch and haven't switched yet
        for i, switch_epoch in enumerate(self.switch_epochs):
            if current_epoch == switch_epoch and i != self.current_timestep_index:
                return True
        
        return False
    
    def update_timesteps(self, current_epoch: int) -> Optional[int]:
        """
        Update the current timestep setting if needed.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
            
        Returns:
            New timestep value if switched, None if no change
        """
        if not self.should_switch_timesteps(current_epoch):
            return None
        
        # Find the appropriate timestep index for this epoch
        new_index = None
        for i, switch_epoch in enumerate(self.switch_epochs):
            if current_epoch == switch_epoch:
                new_index = i
                break
        
        if new_index is not None and new_index != self.current_timestep_index:
            old_timesteps = self.current_timesteps
            self.current_timestep_index = new_index
            self.current_timesteps = self.timesteps[new_index]
            
            info(f"Switching timesteps at epoch {current_epoch}: {old_timesteps} -> {self.current_timesteps}")
            return self.current_timesteps
        
        return None
    
    def get_current_timesteps(self) -> int:
        """Get the current number of timesteps."""
        return self.current_timesteps
    
    def get_max_timesteps(self) -> int:
        """Get the maximum number of timesteps that will be used."""
        return max(self.timesteps) if self.timesteps else 1
    
    def is_enabled(self) -> bool:
        """Check if progressive timestep training is enabled."""
        return self.enabled
    
    def get_progression_info(self) -> Dict[str, Any]:
        """Get information about the timestep progression."""
        return {
            'enabled': self.enabled,
            'timesteps': self.timesteps,
            'switch_epochs': self.switch_epochs,
            'current_timesteps': self.current_timesteps,
            'current_index': self.current_timestep_index,
            'max_timesteps': self.get_max_timesteps()
        }
