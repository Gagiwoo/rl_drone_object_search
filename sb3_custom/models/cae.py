"""
Convolutional AutoEncoder for Global Map Dimensionality Reduction
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class GlobalMapCAE(nn.Module):
    """
    Improved Convolutional AutoEncoder for compressing global map
    with BatchNorm and deeper architecture for better reconstruction
    
    Input: (B, C, H, W) = (batch, 3, 32, 32)
    Latent: (B, latent_dim)
    Output: (B, 3, 32, 32)
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        latent_dim: int = 512
    ):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Encoder: (3, 32, 32) → latent_dim
        # 더 깊고 강력한 구조 with BatchNorm
        self.encoder = nn.Sequential(
            # Layer 1: 32x32 → 16x16
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 2: 16x16 → 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 3: 8x8 → 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Flatten and compress to latent
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim)
        )
        
        # Decoder: latent_dim → (3, 32, 32)
        # Encoder와 대칭적인 구조
        self.decoder = nn.Sequential(
            # Expand from latent
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            
            # Layer 1: 4x4 → 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 2: 8x8 → 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3: 16x16 → 32x32
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # [0, 1] range for normalized input
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode and decode
        
        Returns:
            reconstructed: Reconstructed input
            latent: Latent representation
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
