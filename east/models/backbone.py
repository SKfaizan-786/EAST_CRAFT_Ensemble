"""
ResNet Backbone for EAST

Implements a modular ResNet-18 backbone for feature extraction.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    """
    ResNet-18 backbone for EAST feature extraction.
    Returns feature maps from multiple stages for feature fusion.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # Initial layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        # Residual blocks
        self.layer1 = resnet.layer1  # Output stride 4
        self.layer2 = resnet.layer2  # Output stride 8
        self.layer3 = resnet.layer3  # Output stride 16
        self.layer4 = resnet.layer4  # Output stride 32
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Feature stages
        f1 = self.layer1(x)  # 1/4 resolution
        f2 = self.layer2(f1) # 1/8 resolution
        f3 = self.layer3(f2) # 1/16 resolution
        f4 = self.layer4(f3) # 1/32 resolution
        return f1, f2, f3, f4
