"""
EAST Model Integration

Combines backbone, feature fusion, and output heads into a single model.
"""

import torch
import torch.nn as nn
from .backbone import ResNetBackbone
from .feature_fusion import FeatureFusionNetwork
from .output_heads import EASTOutputHeads

class EAST(nn.Module):
    """
    Complete EAST model: backbone + fusion + output heads.
    """
    def __init__(self, backbone_pretrained=True):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=backbone_pretrained)
        self.fusion = FeatureFusionNetwork()
        self.heads = EASTOutputHeads()
    
    def forward(self, x):
        # Extract multi-scale features
        f1, f2, f3, f4 = self.backbone(x)
        # Fuse features
        fused = self.fusion(f1, f2, f3, f4)
        # Predict score and geometry maps
        score, geometry = self.heads(fused)
        return score, geometry
