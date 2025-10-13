"""
Test EAST Model Forward Pass

Verifies output shapes and basic functionality of the EAST model.
"""

import torch
from east.models.east_model import EAST

def test_east_model_forward():
    print("Testing EAST model forward pass...")
    # Create model
    model = EAST(backbone_pretrained=False)
    model.eval()
    # Dummy input (batch_size=2, 3 channels, 256x256)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        score, geometry = model(x)
    print(f"Score map shape: {score.shape}")
    print(f"Geometry map shape: {geometry.shape}")
    # Expected output shapes
    assert score.shape == (2, 1, 64, 64), "Score map shape mismatch"
    assert geometry.shape == (2, 5, 64, 64), "Geometry map shape mismatch"
    print("EAST model forward pass test passed!")

if __name__ == "__main__":
    test_east_model_forward()
