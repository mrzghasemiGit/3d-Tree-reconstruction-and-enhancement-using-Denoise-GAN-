# 3D Tree Reconstruction Pipeline

A two-stage system for processing and enhancing 3D tree structure data

## Features
- **Stage 1: point cloud Fusion** - Aligns above/below canopy point clouds using stable Geometric features
- **Stage 2: Structural Enhancement** - Denoise-GAN reconstructs trees using an unsupervised GAN, mitigating outliers through its loss function. 

## Installation

### Requirements
- Python 3.8+
- NVIDIA GPU (Recommended for Stage 2)

```bash
git clone https://github.com/yourusername/TreeProcessing.git
cd Tree 3D reconstruction
pip install -r requirements.txt