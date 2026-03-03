"""
Multi-Channel CAE: Each channel processed independently
Input: (3, 32, 32) → 3 separate CAEs → concat latents
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple


class SingleChannelCAE(nn.Module):
    """
    CAE for single channel (32×32 grayscale)
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: (1, 32, 32) → latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # → (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # → (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # → (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )
        
        # Decoder: latent_dim → (1, 32, 32)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single channel"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode single channel"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class MultiChannelCAE(nn.Module):
    """
    Multi-Channel CAE: 3 independent CAEs for each channel
    
    Input: (B, 3, 32, 32)
    Output: 3 × latent_dim (e.g., 3 × 128 = 384)
    """
    def __init__(self, latent_dim_per_channel: int = 128):
        super().__init__()
        self.latent_dim_per_channel = latent_dim_per_channel
        self.total_latent_dim = 3 * latent_dim_per_channel
        
        # 3개의 독립적인 CAE (각 채널별)
        self.cae_channel_0 = SingleChannelCAE(latent_dim_per_channel)
        self.cae_channel_1 = SingleChannelCAE(latent_dim_per_channel)
        self.cae_channel_2 = SingleChannelCAE(latent_dim_per_channel)
        
        print(f"✓ MultiChannelCAE initialized:")
        print(f"  - Latent dim per channel: {latent_dim_per_channel}")
        print(f"  - Total latent dim: {self.total_latent_dim}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode 3-channel input separately
        
        Args:
            x: (B, 3, 32, 32)
        
        Returns:
            latent: (B, 3 × latent_dim)
        """
        # 각 채널 분리
        channel_0 = x[:, 0:1, :, :]  # (B, 1, 32, 32)
        channel_1 = x[:, 1:2, :, :]
        channel_2 = x[:, 2:3, :, :]
        
        # 각 채널별 인코딩
        latent_0 = self.cae_channel_0.encode(channel_0)
        latent_1 = self.cae_channel_1.encode(channel_1)
        latent_2 = self.cae_channel_2.encode(channel_2)
        
        # Concatenate
        latent = torch.cat([latent_0, latent_1, latent_2], dim=1)
        return latent
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode concatenated latent back to 3 channels
        
        Args:
            z: (B, 3 × latent_dim)
        
        Returns:
            reconstructed: (B, 3, 32, 32)
        """
        # Latent 분리
        latent_0 = z[:, :self.latent_dim_per_channel]
        latent_1 = z[:, self.latent_dim_per_channel:2*self.latent_dim_per_channel]
        latent_2 = z[:, 2*self.latent_dim_per_channel:]
        
        # 각 채널별 디코딩
        recon_0 = self.cae_channel_0.decode(latent_0)
        recon_1 = self.cae_channel_1.decode(latent_1)
        recon_2 = self.cae_channel_2.decode(latent_2)
        
        # Concatenate
        reconstructed = torch.cat([recon_0, recon_1, recon_2], dim=1)
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (B, 3, 32, 32)
        
        Returns:
            reconstructed: (B, 3, 32, 32)
            latent: (B, 3 × latent_dim)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
