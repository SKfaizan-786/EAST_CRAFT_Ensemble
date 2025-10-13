"""
EAST Output Heads

Implements score map and geometry map prediction heads.
"""

import torch
import torch.nn as nn

class EASTOutputHeads(nn.Module):
    """
    Output heads for EAST: score map and geometry map.
    - Score map: 1 channel (text/non-text)
    - Geometry map: 5 channels (RBOX: top, right, bottom, left, angle)
    """
    def __init__(self, in_channels=32):
        super().__init__()
        # Score map head
        self.score_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        # Geometry map head
        self.geometry_head = nn.Conv2d(in_channels, 5, 1)
    
    def forward(self, x):
        score = self.score_head(x)
        geometry = self.geometry_head(x)
        return score, geometry
