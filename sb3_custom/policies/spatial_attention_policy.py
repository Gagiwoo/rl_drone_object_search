"""
Custom DQN Policy with Spatial Attention Feature Extractor
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch as th
from stable_baselines3.dqn.policies import DQNPolicy

from sb3_custom.models.spatial_attention_extractor import (
    SpatialAttentionExtractor,
    LearnablePoolingExtractor
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor 


if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    import gymnasium as gym
    from stable_baselines3.common.type_aliases import Schedule


class SpatialAttentionPolicy(DQNPolicy):
    """
    DQN Policy with Spatial Attention + Multi-Scale Feature Extractor
    
    Features:
    - CBAM-style Channel and Spatial Attention
    - Residual connections
    - Hierarchical multi-scale features
    - Returns Q-values for logging
    """
    
    def __init__(
        self,
        observation_space: "gym.Space",
        action_space: "gym.Space",
        lr_schedule: "Schedule",
        net_arch: list[int] | None = None,
        activation_fn: type[th.nn.Module] = th.nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = SpatialAttentionExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
    
    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Get action and Q-values from observation
        
        Returns:
            action: Selected action
            q_values: Q-values for all actions
            state: RNN state (None for DQN)
        """
        self.set_training_mode(False)
        
        observation, vectorized_env = self.obs_to_tensor(observation)
        
        with th.no_grad():
            q_values = self.q_net(observation)
            
            # Greedy action
            action = q_values.argmax(dim=1).reshape(-1)
            
            # Convert to numpy
            action_np = action.cpu().numpy()
            q_values_np = q_values.cpu().numpy()
        
        # Convert to correct action space type
        if self.action_space.__class__.__name__ == "Discrete":
            action_np = action_np.astype(int)
        
        return action_np, q_values_np, state


class LearnablePoolingPolicy(DQNPolicy):
    """
    Baseline Policy with Learnable Pooling (for comparison)
    """
    
    def __init__(
        self,
        observation_space: "gym.Space",
        action_space: "gym.Space",
        lr_schedule: "Schedule",
        net_arch: list[int] | None = None,
        activation_fn: type[th.nn.Module] = th.nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = LearnablePoolingExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
    
    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...] | None]:
        """Same as SpatialAttentionPolicy"""
        self.set_training_mode(False)
        
        observation, vectorized_env = self.obs_to_tensor(observation)
        
        with th.no_grad():
            q_values = self.q_net(observation)
            action = q_values.argmax(dim=1).reshape(-1)
            action_np = action.cpu().numpy()
            q_values_np = q_values.cpu().numpy()
        
        if self.action_space.__class__.__name__ == "Discrete":
            action_np = action_np.astype(int)
        
        return action_np, q_values_np, state
