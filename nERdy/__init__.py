"""
nERdy: Image processing based method for endoplasmic reticulum segmentation.

This module provides morphological image processing operations for the 
segmentation of tubular structures from input ER samples.

Requirements:
- MATLAB R2021a or later with Image Processing Toolbox
- MATLAB Engine API for Python

Usage:
    python nerdy_runner.py path/to/image.png
"""

from .nerdy_runner import preprocess, runner

__version__ = "1.0.0"
__all__ = ["preprocess", "runner"]
