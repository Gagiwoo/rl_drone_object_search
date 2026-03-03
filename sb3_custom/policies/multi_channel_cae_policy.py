"""
DQN Policy with Multi-Channel CAE (channel-wise processing)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np
from stable_baselines3.dqn.policies import DQNPolicy
from gymnasium import spaces
from sb3_custom.models.multi_channel_cae import MultiChannelCAE


class MultiChannelCAEFeaturesExtractor(nn.Module):
    """
    Feature extractor using Multi-Channel CAE
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        cae_model_path: str,
        latent_dim_per_channel: int = 128,
        freeze_cae: bool = True,
        config: dict = None
    ):
        super().__init__()
        
        # Load Multi-Channel CAE
        global_map_shape = observation_space['global_map'].shape
        self.cae = MultiChannelCAE(latent_dim_per_channel=latent_dim_per_channel)
        self.cae.load_state_dict(torch.load(cae_model_path))
        
        if freeze_cae:
            self.cae.eval()
            for param in self.cae.parameters():
                param.requires_grad = False
            print(f"✓ Multi-Channel CAE loaded (frozen)")
        
        print(f"  - Latent per channel: {latent_dim_per_channel}")
        print(f"  - Total latent: {self.cae.total_latent_dim}")
        
        # Local map CNN
        if config and 'local_map' in config:
            conv_config = config['local_map']
            kernel_size = conv_config.get('conv_kernel_size', 5)
            num_kernels = conv_config.get('conv_kernels_num', 16)
            num_layers = conv_config.get('conv_layer_num', 2)
        else:
            kernel_size = 5
            num_kernels = 16
            num_layers = 2
        
        layers = []
        in_channels = observation_space['local_map'].shape[0]
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, num_kernels, kernel_size=kernel_size, stride=1),
                nn.ReLU()
            ])
            in_channels = num_kernels
        layers.append(nn.Flatten())
        self.local_conv = nn.Sequential(*layers)
        
        # Flying time
        self.flying_time_flatten = nn.Flatten()
        
        # Calculate dimensions
        with torch.no_grad():
            sample_obs = {
                'global_map': torch.zeros((1, *global_map_shape)),
                'local_map': torch.zeros((1, *observation_space['local_map'].shape)),
                'flying_time': torch.zeros((1, *observation_space['flying_time'].shape))
            }
            features = self._forward(sample_obs)
            self._features_dim = features.shape[1]
        
        print(f"✓ Total feature dim: {self._features_dim}")
    
    def _forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Global map → Multi-Channel CAE
        global_latent = self.cae.encode(observations['global_map'])
        
        # Local map → CNN
        local_features = self.local_conv(observations['local_map'])
        
        # Flying time
        flying_time = self.flying_time_flatten(observations['flying_time'])
        
        return torch.cat([global_latent, local_features, flying_time], dim=1)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward(observations)
    
    @property
    def features_dim(self) -> int:
        return self._features_dim


class MultiChannelCAEPolicy(DQNPolicy):
    """DQN Policy with Multi-Channel CAE"""
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule,
        softmax_scaling: Optional[float] = None,
        **kwargs
    ):
        self.softmax_scaling = softmax_scaling
        features_kwargs = kwargs.pop('features_extractor_kwargs', {})
        kwargs['features_extractor_class'] = MultiChannelCAEFeaturesExtractor
        kwargs['features_extractor_kwargs'] = features_kwargs
        
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
    
    def predict(
        self,
        observation: np.ndarray | Dict[str, np.ndarray],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        action, state = super().predict(observation, state, episode_start, deterministic)
        
        self.set_training_mode(False)
        observation, vectorized_env = self.obs_to_tensor(observation)
        
        with torch.no_grad():
            q_values = self.q_net(observation)
            action_values = q_values.cpu().numpy()
        
        return action, action_values, state
