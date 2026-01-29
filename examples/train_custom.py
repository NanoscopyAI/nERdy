#!/usr/bin/env python
"""
Example script demonstrating nERdy+ training on a custom dataset.

Usage:
    python examples/train_custom.py --data /path/to/dataset --epochs 100

This script shows how to train the nERdy+ model on your own ER dataset.
"""

import argparse
import os
import sys

# Add nERdy+ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nERdy+'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from model import D4nERdy
from dataloader import ERDataset
from optimizer import VectorAdam


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for inputs, masks in pbar:
        inputs, masks = inputs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def main():
    parser = argparse.ArgumentParser(
        description='Train nERdy+ on a custom dataset'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='trained_model.pth',
        help='Path to save trained model (default: trained_model.pth)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--split',
        type=float,
        default=0.8,
        help='Train/test split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Check dataset
    if not os.path.exists(args.data):
        print(f"Error: Dataset directory not found: {args.data}")
        sys.exit(1)
    
    # Create dataset
    print(f"Loading dataset from: {args.data}")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ERDataset(args.data, transform=transform)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Split dataset
    train_size = int(args.split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("Creating D4nERdy model...")
    model = D4nERdy(in_channels=1, out_channels=1).to(device)
    
    # Create loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = VectorAdam(
        [{'params': model.parameters(), 'axis': -1}],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_test_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_loss = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved best model (loss: {test_loss:.4f})")
    
    print(f"\nTraining complete! Best model saved to: {args.output}")
    print(f"Best test loss: {best_test_loss:.4f}")


if __name__ == '__main__':
    main()
