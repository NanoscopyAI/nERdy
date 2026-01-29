#!/usr/bin/env python
"""
Example script demonstrating nERdy+ inference on a sample image.

Usage:
    python examples/run_inference.py --input path/to/image.png --output path/to/output.png

This script loads the pre-trained nERdy+ model and performs ER segmentation
on the provided input image.
"""

import argparse
import os
import sys

# Add nERdy+ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nERdy+'))

import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from model import D4nERdy
from postprocessing import postprocessing


def load_model(model_path: str, device: torch.device) -> D4nERdy:
    """Load the pre-trained nERdy+ model."""
    model = D4nERdy(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """Load and preprocess an image for inference."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def run_inference(model: D4nERdy, image_tensor: torch.Tensor) -> np.ndarray:
    """Run inference and return the segmentation mask."""
    with torch.no_grad():
        output = model(image_tensor)
        prob_map = torch.sigmoid(output).cpu().squeeze().numpy()
    
    # Normalize to [0, 1]
    prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-8)
    
    # Apply post-processing
    segmentation = postprocessing(prob_map)
    
    return segmentation


def save_result(segmentation: np.ndarray, output_path: str):
    """Save the segmentation result."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to uint8 for saving
    seg_uint8 = segmentation.astype(np.uint8)
    Image.fromarray(seg_uint8).save(output_path)
    print(f"Segmentation saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run nERdy+ inference on an ER image'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input image (PNG, TIFF, etc.)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save output segmentation (default: input_segmentation.png)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to model weights (default: nERdy+/NNet_groupy_p4m_v2_VecAdam.pth)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for inference (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Welcome message
    print("\n" + "="*50)
    print("nERdy+ Inference Script")
    print("="*50 + "\n")
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using CPU (this may be slower)")
    
    # Set default model path
    if args.model is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.model = os.path.join(script_dir, '..', 'nERdy+', 'NNet_groupy_p4m_v2_VecAdam.pth')
    
    # Set default output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_segmentation.png"
    
    # Check if files exist
    if not os.path.exists(args.input):
        print(f"\nError: Input file not found!")
        print(f"    Looked for: {args.input}")
        print(f"\n    Tips:")
        print(f"    - Check if the file path is correct")
        print(f"    - Use the full path: /Users/you/images/file.png")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"\nError: Model file not found!")
        print(f"    Looked for: {args.model}")
        print(f"\n    Tips:")
        print(f"    - Make sure you're running from the nERdy folder")
        print(f"    - The pre-trained model should be in nERdy+/")
        sys.exit(1)
    
    print(f"Loading model...")
    model = load_model(args.model, device)
    
    print(f"Processing image: {os.path.basename(args.input)}")
    image_tensor = preprocess_image(args.input, device)
    
    print("Running neural network...")
    segmentation = run_inference(model, image_tensor)
    
    save_result(segmentation, args.output)
    print("\n" + "="*50)
    print("Done! Segmentation complete.")
    print(f"Result saved to: {args.output}")
    print("="*50)
    print("\nTip: Open the output image to see white=ER, black=background")


if __name__ == '__main__':
    main()
