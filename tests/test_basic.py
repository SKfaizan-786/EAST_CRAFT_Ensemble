"""
Test configuration and imports
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

@pytest.fixture
def sample_image():
    """Generate a sample image for testing"""
    return torch.randn(3, 512, 512)

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'model': {
            'backbone': 'resnet18',
            'pretrained': False,
            'feature_dim': 128
        },
        'training': {
            'batch_size': 2,
            'learning_rate': 0.001
        }
    }

def test_imports():
    """Test that core imports work"""
    try:
        import east
        assert hasattr(east, '__version__')
    except ImportError:
        pytest.skip("EAST package not yet implemented")

def test_torch_installation():
    """Test PyTorch installation and CUDA availability"""
    assert torch.__version__ is not None
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")

def test_sample_fixtures(sample_image, sample_config):
    """Test that fixtures work correctly"""
    assert sample_image.shape == (3, 512, 512)
    assert sample_config['model']['backbone'] == 'resnet18'