"""Learning Rate Schedules for Stable Training"""

import math


def warmup_cosine_schedule(initial_value: float, warmup_steps: int = 50000):
    """
    Learning Rate Schedule with Warmup + Cosine Decay
    
    Phase 1 (Warmup): 0 → initial_value (linearly)
    Phase 2 (Decay): initial_value → initial_value * 0.1 (cosine)
    
    Args:
        initial_value: Peak learning rate
        warmup_steps: Number of warmup steps
    
    Returns:
        Learning rate schedule function
    """
    def func(progress_remaining: float) -> float:
        """
        Args:
            progress_remaining: 1.0 (start) → 0.0 (end)
        Returns:
            Current learning rate
        """
        total_steps = 2000000  # Match n_timesteps
        current_step = int((1 - progress_remaining) * total_steps)
        
        if current_step < warmup_steps:
            # Warmup: Linear increase
            warmup_progress = current_step / warmup_steps
            return initial_value * warmup_progress
        else:
            # Cosine decay
            decay_steps = total_steps - warmup_steps
            decay_progress = (current_step - warmup_steps) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
            
            # Decay to 10% of initial
            return initial_value * (0.1 + 0.9 * cosine_decay)
    
    return func


def linear_schedule(initial_value: float):
    """
    Simple linear decay
    """
    def func(progress_remaining: float) -> float:
        return initial_value * progress_remaining
    
    return func
