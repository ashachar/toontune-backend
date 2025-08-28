"""
Timing control for burn animation effects.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class BurnTiming:
    """Controls timing parameters for burn animation."""
    
    # Phase durations (in seconds)
    stable_duration: float = 0.1      # Initial stable display
    ignite_duration: float = 0.2      # Ignition/starting to burn
    burn_duration: float = 0.8        # Main burning phase
    smoke_duration: float = 0.5       # Smoke rising and dispersing
    
    # Stagger parameters
    burn_stagger: float = 0.1         # Delay between letters starting to burn
    reverse_order: bool = False       # Burn from last to first letter
    random_order: bool = False        # Random burn order
    
    def get_letter_phase(
        self, 
        frame: int, 
        letter_index: int, 
        total_letters: int,
        fps: int
    ) -> Tuple[str, float]:
        """
        Get current phase and progress for a letter.
        
        Returns:
            Tuple of (phase_name, phase_progress)
            Phases: 'stable', 'ignite', 'burn', 'smoke', 'gone'
        """
        # Calculate when this letter starts burning
        if self.random_order:
            # Use deterministic "random" based on letter index
            np.random.seed(letter_index + 42)
            stagger_offset = np.random.uniform(0, self.burn_stagger * (total_letters - 1))
        elif self.reverse_order:
            stagger_offset = (total_letters - 1 - letter_index) * self.burn_stagger
        else:
            stagger_offset = letter_index * self.burn_stagger
        
        current_time = frame / fps
        letter_start_time = stagger_offset
        time_since_start = current_time - letter_start_time
        
        # Determine phase and progress
        if time_since_start < 0:
            return 'waiting', 0.0
        
        if time_since_start < self.stable_duration:
            progress = time_since_start / max(self.stable_duration, 0.01)
            return 'stable', progress
        
        time_since_stable = time_since_start - self.stable_duration
        
        if time_since_stable < self.ignite_duration:
            progress = time_since_stable / max(self.ignite_duration, 0.01)
            return 'ignite', progress
        
        time_since_ignite = time_since_stable - self.ignite_duration
        
        if time_since_ignite < self.burn_duration:
            progress = time_since_ignite / max(self.burn_duration, 0.01)
            return 'burn', progress
        
        time_since_burn = time_since_ignite - self.burn_duration
        
        if time_since_burn < self.smoke_duration:
            progress = time_since_burn / max(self.smoke_duration, 0.01)
            return 'smoke', progress
        
        return 'gone', 1.0
    
    def get_total_duration(self, total_letters: int) -> float:
        """Get total animation duration."""
        stagger_total = self.burn_stagger * (total_letters - 1)
        phase_total = (
            self.stable_duration + 
            self.ignite_duration + 
            self.burn_duration + 
            self.smoke_duration
        )
        return stagger_total + phase_total