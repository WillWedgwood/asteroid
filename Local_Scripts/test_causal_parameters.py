import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from asteroid.data.librimix_dataset import LibriMix
from torch.utils.data import DataLoader
from asteroid.models import ConvTasNet
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr, PairwiseNegSDR
import torch
import torchaudio
import csv
import numpy as np
import time
from util import get_path
import matplotlib.pyplot as plt
import datetime
import torch.nn as nn
from itertools import product

def check_sample_rate(file_path, expected_sample_rate):
    """Checks if the audio file has the expected sample rate."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != expected_sample_rate:
        raise ValueError(f"Sample rate mismatch in file {file_path}. "
                         f"Expected {expected_sample_rate}, but got {sample_rate}.")

def train_epoch(model, train_loader, optimizer, criterion, device, clip_value):
    model.train()
    total_loss = 0

    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main(sample_rate, metadata_path, num_epochs, hyperparams, results_path, epoch_results_path):

    # Load metadata from CSV file
    metadata_df = pd.read_csv(metadata_path)

    # Check the sample rate of the first two clean and mixed files
    mixed_file = metadata_df.iloc[0]['mixture_path']
    clean_file = metadata_df.iloc[0]['source_1_path']

    # Perform sample rate checks
    try:
        check_sample_rate(clean_file, sample_rate)
        check_sample_rate(mixed_file, sample_rate)
    except ValueError as e:
        print(e)
        return

    # Split the data into train+val and test sets
    train_val_df, test_df = train_test_split(metadata_df, test_size=0.2, random_state=42)

    # Split the train+val set into separate train and validation sets
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

    # Save split datasets to CSV for reproducibility
    os.makedirs(get_path('Metadata/train_metadata'), exist_ok=True)
    os.makedirs(get_path('Metadata/val_metadata'), exist_ok=True)
    os.makedirs(get_path('Metadata/test_metadata'), exist_ok=True)
    train_df.to_csv(get_path('Metadata/train_metadata/train_metadata_single.csv'), index=False)
    val_df.to_csv(get_path('Metadata/val_metadata/val_metadata_single.csv'), index=False)
    test_df.to_csv(get_path('Metadata/test_metadata/test_metadata_single.csv'), index=False)

    # Create LibriMix datasets
    train_dataset = LibriMix(csv_dir=get_path("Metadata/train_metadata"), task='enh_single', sample_rate=sample_rate, n_src=1, segment=3)
    val_dataset = LibriMix(csv_dir=get_path('Metadata/val_metadata'), task='enh_single', sample_rate=sample_rate, n_src=1, segment=3)

    # Define the batch size
    batch_size = 4

    # Create DataLoaders for train and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Device configuration
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # Initialize results list
    results = []
    epoch_results = []

    # Iterate over all combinations of hyperparameters
    for lr, wd in product(hyperparams['learning_rate'], hyperparams['weight_decay']):
        print(f"Testing hyperparameters: learning_rate={lr}, weight_decay={wd}")

        # Model definition
        model = ConvTasNet(n_src=1, causal=True).to(device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # Gradient clipping
        clip_value = 1.0

        # Training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs} for hyperparameters: learning_rate={lr}, weight_decay={wd}")

            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip_value)
            val_loss = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            # Save epoch results
            epoch_results.append({
                'learning_rate': lr,
                'weight_decay': wd,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

        # Save final results for this hyperparameter combination
        results.append({
            'learning_rate': lr,
            'weight_decay': wd,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)

    # Save epoch results to CSV
    epoch_results_df = pd.DataFrame(epoch_results)
    epoch_results_df.to_csv(epoch_results_path, index=False)

if __name__ == '__main__':
    sample_rate = 8000
    num_epochs = 10

    # Small dataset for quick testing
    metadata_path = get_path('Metadata/metadata_dev_8k.csv')

    # Hyperparameters to tune
    hyperparams = {
        'learning_rate': [1e-4, 1e-5],
        'weight_decay': [1e-4, 1e-6]
    }

    # Paths to save results
    results_path = 'hyperparameter_tuning_results_dev.csv'
    epoch_results_path = 'epoch_results_dev.csv'

    main(sample_rate, metadata_path, num_epochs, hyperparams, results_path, epoch_results_path)