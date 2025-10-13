"""
EAST-Implement Setup Script
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "EAST-Implement: PyTorch Scene Text Detection"

# Read requirements
def read_requirements():
    # Use minimal requirements for initial setup
    return [
        "torch>=2.1.0",
        "torchvision>=0.16.0", 
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "PyYAML>=6.0.0",
        "tqdm>=4.60.0",
        "pytest>=7.0.0"
    ]

setup(
    name="east-implement",
    version="0.1.0",
    author="Faizan", 
    author_email="your.email@example.com",
    description="PyTorch implementation of EAST for scene text detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/SKfaizan-786/EAST_FYP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0", 
            "black>=23.9.1",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.6.1",
        ],
        "serve": [
            "fastapi>=0.103.2",
            "uvicorn>=0.23.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "east-train=tools.train:main",
            "east-eval=tools.eval:main", 
            "east-infer=tools.infer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "east": ["configs/*.yaml"],
    },
)