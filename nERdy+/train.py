import argparse
import configparser
import os

import torch
from torchvision import transforms

from model import D4nERdy
from dataloader import ERDataset
from optimizer import VectorAdam
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

def main(config):
    torch.manual_seed(config['seed'])

    in_channels = config['in_channels']
    out_channels = config['out_channels']
    model = D4nERdy(in_channels, out_channels)

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    optimizer = VectorAdam([{'params': model.parameters(), 'axis': -1}], lr=config['learning_rate'], betas=config['betas'], eps=config['eps'])

    transform = transforms.Compose([transforms.ToTensor()])
    root_dir = config['dataset_path']
    dataset = ERDataset(root_dir, transform=transform)

    train_size = int(config['split_ratio'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = config['epochs']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item()}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
    average_test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {average_test_loss}")

    torch.save(model.state_dict(), config['save_path'])

def parse_config(file_path):
    parser = configparser.ConfigParser()
    parser.read(file_path)
    return {
        'seed': int(parser['TRAIN']['seed']),
        'in_channels': int(parser['MODEL']['in_channels']),
        'out_channels': int(parser['MODEL']['out_channels']),
        'learning_rate': float(parser['TRAIN']['lr']),
        'betas': tuple(map(float, parser['TRAIN']['betas'].split(','))),
        'eps': float(parser['TRAIN']['eps']),
        'dataset_path': parser['DATA']['dataset_path'],
        'split_ratio': float(parser['TRAIN']['split_ratio']),
        'batch_size': int(parser['TRAIN']['bs']),
        'epochs': int(parser['TRAIN']['epochs']),
        'save_path': parser['MODEL']['save_path']
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, help='Path to config file', default='config.ini')
    args = parser.parse_args()

    config = parse_config(args.config)
    main(config)
