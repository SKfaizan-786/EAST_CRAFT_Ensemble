"""
Feature Fusion Module for EAST

Implements U-shaped feature fusion to merge multi-scale features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusionNetwork(nn.Module):
    """
    U-shaped feature fusion for EAST.
    Fuses features from backbone stages (f1, f2, f3, f4).
    """
    def __init__(self, in_channels=[64, 128, 256, 512], out_channels=32):
        super().__init__()
        # Reduce channels for each stage
        self.reduce1 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.reduce2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.reduce3 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.reduce4 = nn.Conv2d(in_channels[3], out_channels, 1)
        # Final fusion conv
        self.fuse = nn.Conv2d(out_channels * 4, out_channels, 1)
    
    def forward(self, f1, f2, f3, f4):
        # Reduce channels
        r1 = self.reduce1(f1)
        r2 = self.reduce2(f2)
        r3 = self.reduce3(f3)
        r4 = self.reduce4(f4)
        # Upsample to f1 resolution
        r2 = F.interpolate(r2, size=r1.shape[2:], mode='bilinear', align_corners=False)
        r3 = F.interpolate(r3, size=r1.shape[2:], mode='bilinear', align_corners=False)
        r4 = F.interpolate(r4, size=r1.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate
        fused = torch.cat([r1, r2, r3, r4], dim=1)
        # Final fusion
        out = self.fuse(fused)
        return out
