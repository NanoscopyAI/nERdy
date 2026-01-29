# nERdy: network analysis of endoplasmic reticulum dynamics

---

Ashwin Samudre, Guang Gao, Ben Cardoen, Bharat Joshi, Ivan Robert Nabi, Ghassan Hamarneh

Publication: [https://www.nature.com/articles/s42003-025-08892-1](https://www.nature.com/articles/s42003-025-08892-1) (Nature Communications Biology)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**nERdy** is a toolkit for segmenting and analyzing endoplasmic reticulum (ER) networks from microscopy images. It includes:
- **nERdy**: An image processing method
- **nERdy+**: A deep learning method

<img src="figures/nERdy_combined.png" width="800">

---

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start-5-minutes)
2. [Detailed Installation](#detailed-installation)
3. [Using nERdy+ (Deep Learning)](#using-nerdy-deep-learning)
4. [Using nERdy (Image Processing)](#using-nerdy-image-processing)
5. [Training on Your Own Data](#training-on-your-own-data)
6. [Analysis Tools](#analysis-tools)
7. [Troubleshooting](#troubleshooting)
8. [Citation](#citation)

---

## Quick Start

If you just want to try nERdy+ on your images, follow these steps:

### Step 1: Install Conda (if you don't have it)

Download and install Miniconda from: https://docs.anaconda.com/miniconda/install/

**How to check if you have Conda:**
```bash
conda --version
```
If you see a version number (e.g., `conda 23.5.0`), you're good! If you get an error, install Miniconda first.

### Step 2: Download nERdy

```bash
git clone https://github.com/NanoscopyAI/nERdy.git
cd nERdy
```

Or download the ZIP file from GitHub and extract it.

### Step 3: Create Environment and Install

Copy and paste these commands one at a time:

```bash
# Create a new environment (this avoids conflicts with other Python projects)
conda create -n nerdy python=3.10 -y

# Activate the environment
conda activate nerdy

# Install PyTorch (for GPU - recommended if you have an NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR install PyTorch for CPU only (if you don't have a GPU)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

**Expected output after last command:**
```
Successfully installed numpy-1.23.1 pandas-2.0.0 ...
```

### Step 4: Test the Installation

```bash
python -c "import torch; print('PyTorch works!'); print(f'GPU available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch works!
GPU available: True  (or False if using CPU)
```

### Step 5: Run on Your Image

```bash
python examples/run_inference.py --input path/to/your/image.png
```

**What happens:**
- The script loads your image
- Runs the nERdy+ neural network
- Saves a segmentation mask as `your_image_segmentation.png`

---

## Detailed Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU works) | NVIDIA with 4+ GB VRAM |
| Storage | 2 GB | 5 GB |

### For Windows Users

1. Install [Miniconda](https://docs.anaconda.com/miniconda/install/)
2. Open "Anaconda Prompt" from the Start menu
3. Follow the Quick Start steps above

### For Mac Users

1. Install [Miniconda](https://docs.anaconda.com/miniconda/install/)
2. Open Terminal
3. Follow the Quick Start steps above
4. **Note:** Mac does not support CUDA. Use the CPU version of PyTorch.

### For Linux Users

Follow the Quick Start steps in a terminal.

### Verifying Your Installation

Run this test to ensure everything is working:

```bash
# Make sure you're in the nERdy folder and environment is active
conda activate nerdy

# Test imports
python -c "
import torch
import numpy as np
from PIL import Image
print('All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
"
```

---

## Using nERdy+ (Deep Learning)

nERdy+ is a neural network that segments ER structures from microscopy images. **This is the recommended method** for most users as it provides the best results.

### Basic Usage

```bash
# Run on a single image
python examples/run_inference.py --input your_image.png

# Specify output location
python examples/run_inference.py --input your_image.png --output result.png

# Force CPU usage (if GPU causes issues)
python examples/run_inference.py --input your_image.png --device cpu
```

### Input Image Requirements

- **Format:** TIFF
- **Type:** Grayscale (the script converts color images automatically)
- **Size:** Any size works, but 128x128 to 1024x1024 is typical

### Understanding the Output

The output is a binary segmentation mask where:
- **White (255):** ER structure detected
- **Black (0):** Background

### Processing Multiple Images

To process a folder of images, use this Python script:

```python
import os
import subprocess

input_folder = "path/to/your/images"
output_folder = "path/to/results"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.tif', '.tiff', '.jpg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{filename}_seg.png")
        subprocess.run([
            "python", "examples/run_inference.py",
            "--input", input_path,
            "--output", output_path
        ])
        print(f"Processed: {filename}")
```

### Using nERdy+ in Your Own Python Code

```python
import sys
sys.path.append('nERdy+')  # Add nERdy+ to Python path

import torch
from PIL import Image
from torchvision import transforms
from model import D4nERdy
from postprocessing import postprocessing

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = D4nERdy(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('nERdy+/NNet_groupy_p4m_v2_VecAdam.pth', map_location=device))
model = model.to(device)
model.eval()

# Load and process your image
image = Image.open('your_image.png').convert('L')  # Convert to grayscale
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    output = model(image_tensor)
    prob_map = torch.sigmoid(output).cpu().squeeze().numpy()

# Post-process to get binary mask
segmentation = postprocessing(prob_map)

# Save result
Image.fromarray(segmentation.astype('uint8')).save('segmentation.png')
```

---

## Using nERdy (Image Processing)

nERdy is a traditional image processing method that uses morphological operations and vesselness filtering. It requires MATLAB but provides an interpretable pipeline.

### When to Use nERdy vs nERdy+

| Use nERdy when... | Use nERdy+ when... |
|-------------------|-------------------|
| You have MATLAB available | You want best accuracy |
| You want interpretable results | You don't have MATLAB |
| You need to tune parameters manually | You prefer pre-trained models |

### Prerequisites

1. **MATLAB R2021a or later** with Image Processing Toolbox
2. **MATLAB Engine API for Python**

### Step 1: Install MATLAB Engine for Python

First, find your MATLAB installation path:
- **Windows:** `C:\Program Files\MATLAB\R2023a`
- **Mac:** `/Applications/MATLAB_R2023a.app`
- **Linux:** `/usr/local/MATLAB/R2023a`

Then install the MATLAB Engine:

```bash
# Navigate to MATLAB's Python engine folder
cd "<your_matlab_path>/extern/engines/python"

# Install (may need admin/sudo)
python setup.py install
```

**Verify installation:**
```python
python -c "import matlab.engine; print('MATLAB Engine installed successfully!')"
```

### Step 2: Run nERdy

```bash
# Make sure you're in the nERdy folder
cd nERdy

# Run on your image
python nerdy_runner.py path/to/your/image.png
```

### What nERdy Does

The pipeline has two stages:

**Stage 1: Python Preprocessing** (`preprocess` function)
1. Normalizes image to [0, 1] range
2. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Area opening to remove small bright artifacts
4. Erosion to thin structures
5. Local thresholding

**Stage 2: MATLAB Vessel Enhancement** (`Vessel2d.m`)
1. Applies Jerman's vesselness filter at multiple scales (σ = 0.5 to 2.5)
2. Enhances tubular structures
3. Binarizes the result

### Output Files

After running, you'll get:
- `preprocessed_<filename>` - Image after Python preprocessing
- `preprocessed_<filename>_enhance.png` - **Final segmentation result**

### Example

```bash
# If your image is called "er_sample.png"
cd nERdy
python nerdy_runner.py er_sample.png

# Output files:
# - preprocessed_er_sample.png (intermediate)
# - preprocessed_er_sample_enhance.png (final result)
```

### Troubleshooting nERdy

**"No module named 'matlab.engine'"**
- MATLAB Engine is not installed
- Follow Step 1 above to install it

**"MATLAB session cannot be started"**
- Make sure MATLAB is properly installed and licensed
- Try running MATLAB manually first to verify it works

**"Image Processing Toolbox not found"**
- Install the Image Processing Toolbox in MATLAB
- In MATLAB: Home → Add-Ons → Get Add-Ons → Search "Image Processing Toolbox"

---

## Training on Your Own Data

If you have your own annotated ER images, you can train a custom model.

### Step 1: Prepare Your Dataset

Organize your data like this:

```
my_dataset/
    train/
        images/
            image001.png
            image002.png
            ...
        masks/
            image001_mask.png
            image002_mask.png
            ...
```

**Important:**
- Mask filenames must match image filenames with `_mask` added
- Masks should be binary (0 for background, 255 for ER)
- Images should be grayscale

### Step 2: Configure Training

Edit `nERdy+/config.ini`:

```ini
[TRAIN]
seed = 34
lr = 1e-3
betas = 0.9, 0.999
eps = 1e-08
split_ratio = 0.8
bs = 16          # Reduce if you run out of GPU memory
epochs = 100

[MODEL]
in_channels = 1
out_channels = 1
save_path = my_trained_model.pth

[DATA]
dataset_path = /full/path/to/my_dataset
```

### Step 3: Train

```bash
cd nERdy+
python train.py --config config.ini
```

**What to expect:**
- Training progress will print after each epoch
- Training 100 epochs takes ~30 minutes on a GPU, ~2-3 hours on CPU
- The model is saved to the path specified in config.ini

### Step 4: Use Your Trained Model

```bash
python examples/run_inference.py --input test_image.png --model nERdy+/my_trained_model.pth
```

---

## Analysis Tools

nERdy includes tools for analyzing ER network properties.

### Junction Analysis

Analyze junction dynamics from time-series data:

```python
from analysis.junction_analysis import JunctionAnalysis

# Initialize for your microscopy type
ja = JunctionAnalysis('confocal')  # or 'sted'
```

### Plotting Metrics

Visualize segmentation and graph metrics:

```python
from analysis.graph_metrics_plotter import GraphMetricsPlotter
from analysis.segmentation_metrics_plotter import SegmentationMetricsPlotter

# Graph metrics
gmp = GraphMetricsPlotter('confocal')
gmp.plot()

# Segmentation metrics
smp = SegmentationMetricsPlotter()
smp.get_segmentation_perf()
```

<img src="figures/nerdynet-pipeline.drawio.png" width="800">

---

## Data

Download the dataset used in our paper from [Figshare](https://figshare.com/articles/dataset/nERdy_dataset/25241458).

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

**Cause:** PyTorch is not installed or the conda environment is not activated.

**Solution:**
```bash
conda activate nerdy
pip install torch torchvision
```

### "CUDA out of memory"

**Cause:** Your GPU doesn't have enough memory.

**Solutions:**
1. Use CPU instead: `--device cpu`
2. Reduce batch size in config.ini: `bs = 8` or `bs = 4`
3. Process smaller image crops

### "No module named 'model'" or "No module named 'groupy'"

**Cause:** Python can't find the nERdy+ modules.

**Solution:** Make sure you're running from the nERdy root folder:
```bash
cd /path/to/nERdy
python examples/run_inference.py --input image.png
```

### "FileNotFoundError: NNet_groupy_p4m_v2_VecAdam.pth"

**Cause:** The pre-trained model file is missing.

**Solution:** Make sure you downloaded/cloned the complete repository including the .pth file.

### Poor segmentation results

**Possible causes and solutions:**
1. **Image quality:** nERdy+ works best on high-contrast ER images
2. **Image type:** Make sure your image is grayscale microscopy data
3. **Different microscopy:** If results are poor, consider training on your own data

### Mac-specific: "MPS backend not available"

**Cause:** Apple Silicon Macs may have issues with GPU acceleration.

**Solution:** Use CPU mode:
```bash
python examples/run_inference.py --input image.png --device cpu
```

### Still having issues?

1. Check that all requirements are installed: `pip install -r requirements.txt`
2. Try creating a fresh conda environment
3. Open an issue on [GitHub](https://github.com/NanoscopyAI/nERdy/issues) with:
   - Your operating system
   - Error message (full text)
   - Steps you followed

---

## Project Structure

```
nERdy/
├── nERdy/                  # Image processing method (requires MATLAB)
├── nERdy+/                 # Deep learning method (recommended)
│   ├── model.py            # Neural network architecture
│   ├── train.py            # Training script
│   ├── inference.py        # Inference utilities
│   ├── config.ini          # Training configuration
│   └── NNet_groupy_*.pth   # Pre-trained model weights
├── analysis/               # Analysis and visualization tools
├── examples/               # Ready-to-use example scripts
├── test/                   # Unit tests
├── figures/                # Documentation images
└── requirements.txt        # Python dependencies
```

---

## Citation

If you use nERdy in your research, please cite:

```bibtex
@article{samudre2025nerdy,
  title={nERdy: network analysis of endoplasmic reticulum dynamics},
  author={Samudre, Ashwin and Gao, Guang and Cardoen, Ben and Joshi, Bharat and Nabi, Ivan Robert and Hamarneh, Ghassan},
  journal={Communications Biology},
  volume={8},
  number={1},
  pages={1529},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

---

## Acknowledgments

- [GrouPy](https://github.com/adambielski/GrouPy) for group equivariant convolutions
- [PlantCV](https://plantcv.readthedocs.io/) for morphological analysis
- [sknw](https://github.com/Image-Py/sknw) for skeleton to graph conversion
