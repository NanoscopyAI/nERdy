"""
nERdy+: D4-equivariant neural network for endoplasmic reticulum segmentation.

This module provides a D4-equivariant encoder-decoder network trained to 
segment tubular structures from input ER samples.
"""

from .model import D4nERdy, nERdy
from .dataloader import ERDataset
from .postprocessing import postprocessing
from .optimizer import VectorAdam

__version__ = "1.0.0"
__all__ = ["D4nERdy", "nERdy", "ERDataset", "postprocessing", "VectorAdam"]
