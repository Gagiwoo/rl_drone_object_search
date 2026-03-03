"""
Custom Policies for SB3
"""
from sb3_custom.policies.spatial_attention_policy import (
    SpatialAttentionPolicy,
    LearnablePoolingPolicy
)

__all__ = [
    "SpatialAttentionPolicy",
    "LearnablePoolingPolicy",
]
