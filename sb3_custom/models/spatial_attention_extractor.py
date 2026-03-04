"""
Spatial Attention Feature Extractor with Self-Attention and SE Block
- Enhanced CBAM with Self-Attention for long-range dependencies
- SE Block for scale-specific channel recalibration
- Multi-Scale Feature Extraction (Fine/Mid/Coarse)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

if TYPE_CHECKING:
    from typing import Any


# ========================
# Self-Attention Module
# ========================
class SelfAttention(nn.Module):
    """
    Self-Attention Module for capturing long-range spatial dependencies
    - Query-Key-Value attention mechanism
    - Learns relationships between distant regions (e.g., weed clusters)
    - Lightweight with channel reduction (1/8)
    """
    
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        
        # Reduce channels for efficiency
        reduced_channels = max(1, channels // 8)
        
        self.query = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # Learnable weight for gradual application (starts at 0)
        self.gamma = nn.Parameter(th.zeros(1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        print(f"[Self-Attention] Initialized: {channels} channels, "
              f"reduced to {reduced_channels}, dropout={dropout}")
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            out: Self-attended feature map (B, C, H, W)
        """
        B, C, H, W = x.size()
        
        # Query, Key, Value projections
        # query: (B, C', H, W) → (B, H*W, C')
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        # key: (B, C', H, W) → (B, C', H*W)
        key = self.key(x).view(B, -1, H * W)
        # value: (B, C, H, W) → (B, C, H*W)
        value = self.value(x).view(B, -1, H * W)
        
        # Attention map: (B, H*W, H*W)
        # Each position attends to all other positions
        attention = th.bmm(query, key)  # (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values: (B, C, H*W)
        out = th.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable weight
        # gamma starts at 0, gradually increases during training
        out = self.gamma * out + x
        
        return out


# ========================
# Squeeze-and-Excitation Block
# ========================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    - Adaptive channel-wise feature recalibration
    - Squeeze: Global average pooling
    - Excitation: FC layers with sigmoid activation
    - Lightweight and effective
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        reduced_channels = max(1, channels // reduction)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        print(f"[SE Block] Initialized: {channels} channels, "
              f"reduction={reduction} (→{reduced_channels})")
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            out: Recalibrated feature map (B, C, H, W)
        """
        B, C, H, W = x.size()
        
        # Squeeze: Global spatial information into channel descriptor
        squeeze = self.squeeze(x)  # (B, C, 1, 1)
        
        # Excitation: Capture channel-wise dependencies
        excitation = self.excitation(squeeze)  # (B, C, 1, 1)
        
        # Scale: Element-wise multiplication
        return x * excitation


# ========================
# Channel Attention (CBAM)
# ========================
class ChannelAttention(nn.Module):
    """Channel Attention Module (CBAM-style) - Stable version"""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(1, channels // reduction)
        
        # Shared MLP with LayerNorm for stability
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.LayerNorm([channels, 1, 1])
        )
        self.sigmoid = nn.Sigmoid()
        
        print(f"[Channel Attention] Initialized: {channels} channels, "
              f"reduction={reduction} (→{reduced_channels})")
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Clip to prevent overflow
        out = th.clamp(avg_out + max_out, -10, 10)
        out = self.sigmoid(out)
        
        # Prevent zeros (add small epsilon)
        out = out + 1e-8
        
        return x * out


# ========================
# Spatial Attention (CBAM)
# ========================
class SpatialAttention(nn.Module):
    """Spatial Attention Module (CBAM-style) - Stable version"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        print(f"[Spatial Attention] Initialized: kernel_size={kernel_size}")
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        # Channel-wise statistics
        avg_out = th.mean(x, dim=1, keepdim=True)
        max_out, _ = th.max(x, dim=1, keepdim=True)
        
        out = th.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        
        # Clip before sigmoid
        out = th.clamp(out, -10, 10)
        out = self.sigmoid(out)
        
        # Prevent zeros
        out = out + 1e-8
        
        return x * out


# ========================
# Main Feature Extractor
# ========================
class SpatialAttentionExtractor(BaseFeaturesExtractor):
    """
    Enhanced Spatial Attention + Multi-Scale Feature Extractor
    
    Architecture:
        1. Channel Attention (CBAM) - "What channels are important?"
        2. Spatial Attention (CBAM) - "Where are important regions?"
        3. Self-Attention (NEW) - "How do distant regions relate?"
        4. Multi-Scale Convolution with SE Block (NEW)
           - Fine scale (2×2): Individual weeds
           - Mid scale (4×4): Weed clusters
           - Coarse scale (8×8): Overall patterns
        5. Feature Fusion
    
    Args:
        observation_space: Gym observation space (Dict with "global_map")
        attention_enabled: Enable CBAM attention modules
        multi_scale: Enable multi-scale feature extraction
        use_residual: Use residual connections in attention
        attention_strength: Strength of attention (not used currently)
        use_self_attention: Enable Self-Attention module (NEW)
        use_se_block: Enable SE Block in multi-scale conv (NEW)
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        attention_enabled: bool = True,
        multi_scale: bool = True,
        use_residual: bool = True,
        attention_strength: float = 0.5,
        use_self_attention: bool = True,
        use_se_block: bool = True,
    ):
        features_dim = self._calculate_features_dim(
            observation_space, multi_scale
        )
        
        super().__init__(observation_space, features_dim=features_dim)
        
        self.attention_enabled = attention_enabled
        self.multi_scale = multi_scale
        self.use_residual = use_residual
        self.attention_strength = attention_strength
        self.use_self_attention = use_self_attention
        self.use_se_block = use_se_block
        
        global_map_shape = observation_space.spaces["global_map"].shape
        n_input_channels = global_map_shape[0]
        
        print("\n" + "=" * 70)
        print("Initializing Spatial Attention Feature Extractor")
        print("=" * 70)
        
        # ========== Attention Modules ==========
        if self.attention_enabled:
            self.channel_attention = ChannelAttention(
                channels=n_input_channels,
                reduction=max(1, n_input_channels // 4)
            )
            self.spatial_attention = SpatialAttention(kernel_size=7)
            
            # Self-Attention (NEW)
            if self.use_self_attention:
                self.self_attention = SelfAttention(
                    channels=n_input_channels,
                    dropout=0.1
                )
            
            print(f"\n[CBAM Attention] Enabled")
            print(f"  - Channel Attention: ✓")
            print(f"  - Spatial Attention: ✓")
            print(f"  - Self-Attention: {'✓' if use_self_attention else '✗'}")
            print(f"  - Residual Connection: {'✓' if use_residual else '✗'}")
        
        # ========== Multi-Scale Convolutions ==========
        if self.multi_scale:
            # Fine scale (2×2 receptive field)
            self.conv_fine = nn.Sequential(
                nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                SEBlock(64) if use_se_block else nn.Identity()
            )
            
            # Mid scale (4×4 receptive field)
            self.conv_mid = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                SEBlock(128) if use_se_block else nn.Identity()
            )
            
            # Coarse scale (8×8 receptive field)
            self.conv_coarse = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                SEBlock(256) if use_se_block else nn.Identity()
            )
            
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
            
            print(f"\n[Multi-Scale Convolution] Enabled")
            print(f"  - Fine scale: 64 channels (2×2)")
            print(f"  - Mid scale: 128 channels (4×4)")
            print(f"  - Coarse scale: 256 channels (8×8)")
            print(f"  - SE Block: {'✓' if use_se_block else '✗'}")
            print(f"  - Features Dim: {features_dim}")
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(n_input_channels, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
            
            print(f"\n[Single-Scale Convolution] Enabled")
            print(f"  - Features Dim: {features_dim}")
        
        print("=" * 70 + "\n")
    
    def _calculate_features_dim(
        self, 
        observation_space: gym.spaces.Dict, 
        multi_scale: bool
    ) -> int:
        if multi_scale:
            # (64 + 128 + 256) channels × 4 × 4 spatial
            return (64 + 128 + 256) * 4 * 4
        else:
            return 256 * 4 * 4
    
    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        """
        Forward pass with enhanced attention
        
        Args:
            observations: Dict with "global_map" key (B, C, H, W)
        Returns:
            features: Flattened feature vector (B, features_dim)
        """
        global_map = observations["global_map"]
        
        # ✅ Check for NaN in input
        if th.isnan(global_map).any():
            print("WARNING: NaN detected in input global_map!")
            global_map = th.nan_to_num(global_map, nan=0.0)
        
        # ========== 1. Attention Pipeline ==========
        if self.attention_enabled:
            if self.use_residual:
                original = global_map.clone()
            
            # 1.1 Channel Attention
            global_map = self.channel_attention(global_map)
            
            if th.isnan(global_map).any():
                print("WARNING: NaN after channel attention!")
                global_map = th.nan_to_num(global_map, nan=0.0)
            
            # 1.2 Spatial Attention
            global_map = self.spatial_attention(global_map)
            
            if th.isnan(global_map).any():
                print("WARNING: NaN after spatial attention!")
                global_map = th.nan_to_num(global_map, nan=0.0)
            
            # 1.3 Self-Attention (NEW)
            if self.use_self_attention:
                global_map = self.self_attention(global_map)
                
                if th.isnan(global_map).any():
                    print("WARNING: NaN after self-attention!")
                    global_map = th.nan_to_num(global_map, nan=0.0)
            
            # Residual connection
            if self.use_residual:
                global_map = global_map + 0.1 * original
        
        # ========== 2. Multi-Scale Feature Extraction ==========
        if self.multi_scale:
            feat_fine = self.conv_fine(global_map)
            feat_mid = self.conv_mid(feat_fine)
            feat_coarse = self.conv_coarse(feat_mid)
            
            # Resize all to 4×4
            feat_fine_pooled = self.adaptive_pool(feat_fine)
            feat_mid_pooled = self.adaptive_pool(feat_mid)
            feat_coarse_pooled = self.adaptive_pool(feat_coarse)
            
            # Concatenate multi-scale features
            features = th.cat([
                feat_fine_pooled,
                feat_mid_pooled,
                feat_coarse_pooled
            ], dim=1)
        else:
            features = self.conv(global_map)
            features = self.adaptive_pool(features)
        
        # ✅ Final NaN check
        features = features.flatten(start_dim=1)
        if th.isnan(features).any():
            print("WARNING: NaN in final features!")
            features = th.nan_to_num(features, nan=0.0)
        
        return features


# ========================
# Baseline Extractor
# ========================
class LearnablePoolingExtractor(BaseFeaturesExtractor):
    """Baseline: Learnable Pooling (stable version)"""
    
    def __init__(self, observation_space: gym.spaces.Dict):
        features_dim = 256 * 4 * 4
        super().__init__(observation_space, features_dim=features_dim)
        
        global_map_shape = observation_space.spaces["global_map"].shape
        n_input_channels = global_map_shape[0]
        
        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        print("\n" + "=" * 70)
        print("Initializing Learnable Pooling Extractor (Baseline)")
        print(f"  Features Dim: {features_dim}")
        print("=" * 70 + "\n")
    
    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        global_map = observations["global_map"]
        
        # ✅ NaN check
        if th.isnan(global_map).any():
            global_map = th.nan_to_num(global_map, nan=0.0)
        
        features = self.conv(global_map)
        features = features.flatten(start_dim=1)
        
        if th.isnan(features).any():
            features = th.nan_to_num(features, nan=0.0)
        
        return features
