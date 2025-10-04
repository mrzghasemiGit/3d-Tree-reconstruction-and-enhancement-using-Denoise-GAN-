# 3D Tree Reconstruction Denoise GAN

A two-stage system for processing and enhancing 3D tree structure data

## Features
- **Stage 1: point cloud Fusion** - Aligns above/below canopy point clouds using stable geometric features
- **Stage 2: Structural Enhancement** - Denoise-GAN reconstructs trees using an unsupervised GAN, mitigating outliers through its loss function.

## 📁 Project Structure

```bash
Tree-3D-reconstruction/ 
├── core/
│   ├── Fusion.py           # spatial alignment code
│   └── Denoise-GAN.py       # enhancement network
├── configs/
│   ├── params.yaml         # Fusion and parameters
├── utils/
│   ├── downsample.py       # Downsample functions 
│   ├── remove noise.py     # Noise removal functions   
│   └── visualize.py        # Visualization functions
├── data/                   # Sample PLY files
├── scripts/
│   ├── Fusion.py           # End-to-end execution script
│   └── Enhancement.py      # GAN  script
├── requirements.txt        # Python dependencies
└── README.md               # Usage documentation
```

### Requirements
- Python 3.9+
- NVIDIA GPU (Recommended for Stage 2)

## 🔧 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/mrzghasemiGit/3D-tree-reconstruction.git
cd Tree-3D-reconstruction
```

2. **Create and activate a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate
```

3. **Install the required packages:**

```bash
pip install -r requirements.txt
```

### ▶️ Usage
- Make sure your input point clouds above.ply and below.ply are placed in the Tree-3D-reconstruction folder.

1. **Run the Fusion script**
- This fuses the above- and below-canopy point clouds into one model.

```bash

python "Tree 3D reconstruction/scripts/Fusion.py"
```

2. **(Optional) Enhance the fused point cloud**
- You can further enhance the fused model using the enhancement module:

```bash

python "Tree 3D reconstruction/scripts/Enhancement.py"
```
The final output fused_canopy.ply will be saved in the data directory.

### 📝 Citation
This code is part of the paper:

"Geometry-based Point Cloud Fusion of Dual-Layer UAV Photogrammetry and Unsupervised Generative Adversarial Network for 3D Tree Reconstruction in Semi-Arid Forests" https://doi.org/10.1016/j.compag.2025.111024


