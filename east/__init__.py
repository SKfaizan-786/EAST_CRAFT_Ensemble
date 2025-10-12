"""
EAST-Implement: Efficient and Accurate Scene Text Detection

A PyTorch implementation of EAST (Efficient and Accurate Scene Text) detector
for scene text detection with comprehensive training, evaluation, and 
visualization pipeline.

Author: Faizan
GitHub: https://github.com/SKfaizan-786/EAST_FYP
"""

__version__ = "0.1.0"
__author__ = "Faizan"
__email__ = "your.email@example.com"

from .models import EAST
from .datasets import ICDARDataset
from .losses import EASTLoss
from .utils import *

__all__ = [
    "EAST",
    "ICDARDataset", 
    "EASTLoss",
    "__version__",
    "__author__"
]