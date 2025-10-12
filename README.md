# EAST-Implement: PyTorch Scene Text Detection

A comprehensive PyTorch implementation of **EAST (Efficient and Accurate Scene Text)** detector for scene text detection with complete training, evaluation, and deployment pipeline.

## 🎯 Project Overview

This implementation focuses on:
- **Reproducible Research**: Complete environment specifications and experiment tracking
- **Modern PyTorch**: Best practices with mixed precision, distributed training
- **Production Ready**: Docker containers, ONNX export, REST API serving
- **Educational**: Step-by-step tutorials and comprehensive documentation

## 🚀 Quick Start

### 1. Environment Setup

**Option A: Conda (Recommended)**
```bash
git clone https://github.com/SKfaizan-786/EAST_FYP.git
cd EAST_FYP
conda env create -f environment.yml
conda activate east-implement
```

**Option B: pip + virtualenv**
```bash
git clone https://github.com/SKfaizan-786/EAST_FYP.git
cd EAST_FYP
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Package
```bash
pip install -e .
```

### 3. Download ICDAR 2015 Dataset
```bash
python tools/download_dataset.py --dataset icdar2015 --output data/
```

### 4. Train Model
```bash
python tools/train.py --config configs/east_resnet18.yaml
```

## 📊 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| ICDAR 2015 F-score | >77% | 🔄 In Progress |
| Inference Speed | <50ms (RTX 4090) | 🔄 In Progress |
| Training Memory | <8GB VRAM | 🔄 In Progress |
| Test Coverage | >85% | 🔄 In Progress |

## 🏗️ Architecture

```
Input Image (3×512×512)
         ↓
   ResNet Backbone 
    (conv2-conv5)
         ↓
  Feature Fusion Network
  (Progressive Upsampling)
         ↓
    Dual-Head Output
  ┌─────────┬─────────┐
  ↓         ↓         ↓
Score Map  Geometry Map
(1×128×128) (8×128×128)
         ↓
   Post-processing
    (NMS + Decode)
         ↓
  Text Detections
   (Quadrilaterals)
```

## 📁 Project Structure

```
EAST_FYP/
├── east/                   # Main package
│   ├── models/            # Model architecture
│   ├── datasets/          # Data loading
│   ├── losses/            # Loss functions
│   ├── utils/             # Utilities
│   └── evaluation/        # Evaluation tools
├── configs/               # Configuration files
├── tools/                 # Training/evaluation scripts
├── notebooks/             # Educational tutorials
├── tests/                 # Unit tests
├── docker/                # Docker configurations
├── docs/                  # Documentation
├── requirements.txt       # Dependencies
└── setup.py              # Package setup
```

## 🛠️ Development Status

### ✅ Completed (Sprint 1)
- [x] GitHub repository setup
- [x] Project structure and requirements
- [x] Configuration system
- [x] Package setup and initialization

### 🔄 In Progress (Sprint 2)
- [ ] ICDAR dataset loader
- [ ] Data preprocessing pipeline
- [ ] Ground truth map generation
- [ ] Data augmentation

### ⏳ Planned
- [ ] ResNet backbone implementation
- [ ] Feature fusion network
- [ ] Training pipeline
- [ ] Evaluation framework
- [ ] Docker deployment

## 📚 Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Architecture Overview](docs/architecture.md)** - Model design explanation
- **[Training Guide](docs/training.md)** - How to train your own model
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Deployment Guide](docs/deployment.md)** - Production deployment

## 🎓 Educational Notebooks

1. **[Architecture Explanation](notebooks/01_architecture_overview.ipynb)** 
2. **[Training Tutorial](notebooks/02_training_tutorial.ipynb)**
3. **[Evaluation Demo](notebooks/03_evaluation_demo.ipynb)**
4. **[Deployment Example](notebooks/04_deployment_example.ipynb)**

## 🐳 Docker Deployment

**Training Container**
```bash
docker build -f docker/Dockerfile.train -t east-train .
docker run --gpus all -v $(pwd)/data:/workspace/data east-train
```

**Serving Container**
```bash
docker build -f docker/Dockerfile.serve -t east-serve .
docker run -p 8000:8000 east-serve
```

## 🧪 Testing

Run tests with coverage:
```bash
pytest tests/ --cov=east --cov-report=html
```

## 📝 Citation

If you use this implementation in your research, please cite:

```bibtex
@software{east_implement_2025,
  title={EAST-Implement: PyTorch Scene Text Detection},
  author={Faizan},
  year={2025},
  url={https://github.com/SKfaizan-786/EAST_FYP}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 🙏 Acknowledgments

- Original EAST paper: [Zhou et al., CVPR 2017](https://arxiv.org/abs/1704.03155)
- ICDAR 2015 dataset organizers
- PyTorch team for the excellent framework
- OpenCV contributors for computer vision tools

## 📞 Contact

- **Author**: Faizan
- **GitHub**: [@SKfaizan-786](https://github.com/SKfaizan-786)
- **Project**: [EAST_FYP](https://github.com/SKfaizan-786/EAST_FYP)

---

⭐ **Star this repository if you find it helpful!**