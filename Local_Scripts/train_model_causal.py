import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

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
import time  # Import time module for measuring elapsed time
from util import get_path
import matplotlib.pyplot as plt
import datetime
import torch.nn as nn

def check_sample_rate(file_path, expected_sample_rate):
    """Checks if the audio file has the expected sample rate."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != expected_sample_rate:
        raise ValueError(f"Sample rate mismatch in file {file_path}. "
                         f"Expected {expected_sample_rate}, but got {sample_rate}.")

def main(sample_rate, metadata_path, num_epochs):

    # Load metadata from CSV file
    metadata_df = pd.read_csv(metadata_path)

    ### Check Sample Rate ###

    # Check the sample rate of the first two clean and mixed files
    mixed_file = metadata_df.iloc[0]['mixture_path']
    clean_file = metadata_df.iloc[0]['source_1_path']

    # Perform sample rate checks. todo: add mono check.
    try:
        check_sample_rate(clean_file, sample_rate)
        check_sample_rate(mixed_file, sample_rate)
    except ValueError as e:
        print(e)
        return

    ### Test Set Up ###

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
    test_dataset = LibriMix(csv_dir=get_path('Metadata/test_metadata'), task='enh_single', sample_rate=sample_rate, n_src=1, segment=3)

    # Define the batch size
    batch_size = 4
    print(f'Sample rate: {sample_rate:.4f}, Batch Size: {batch_size:.4f}')

    # Create DataLoaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model definition
    model = ConvTasNet(n_src=1, causal=True)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Gradient clipping
    clip_value = 1.0

    ###### Define training, validation, and test loops ######

    # Define the training loop
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

    # Define the validation loop
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

    # Define the test loop
    def test(model, test_loader, criterion, device):
        model.eval()
        total_loss = 0
        max_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                max_test_loss = max(max_test_loss, loss.item())
        return total_loss / len(test_loader), max_test_loss

    ###### Training process ######
    
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Processing using: {device}')

    model.to(device)

    # Initialize arrays to store the loss values and epoch times
    train_losses = []
    val_losses = []
    epoch_times = []

    # Ensure the directory for reports exists
    report_dir = 'Trained_Models/Training_Reports'
    os.makedirs(report_dir, exist_ok=True)  # This creates the directory if it doesn't exist

    # Create a report file with a timestamp
    report_filename = os.path.join(report_dir, f"training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(report_filename, "w") as report_file:
        for epoch in range(num_epochs):

            # Record the start time for the epoch
            epoch_start_time = time.time()

            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip_value)
            val_loss = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            # Calculate the time taken for the epoch
            epoch_time = time.time() - epoch_start_time

            # Append losses and epoch time to the respective arrays
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            epoch_times.append(epoch_time)

            # Prepare the log message
            log_message = (f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
                        f'Validation Loss: {val_loss:.4f}, Epoch Time: {epoch_time:.4f} seconds\n')

            # Print the log message
            print(log_message)

            # Write the log message to the report file
            report_file.write(log_message)

    ### Test Model

    # Testing process
    avg_test_loss, max_test_loss = test(model, test_loader, criterion, device)

    # Optionally, save the model
    torch.save(model.state_dict(), 'causal_test.pth')

    # Print the test result
    log_message = (f'Avg Test Loss: {avg_test_loss:.4f}, Max Abs Test Loss: {max_test_loss:.4f}')
    print(log_message)
    #report_file.write(log_message)

    epochs = range(1, len(train_losses) + 1)
    
    # Plotting Train and Validation Losses
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Over Epochs')
    plt.legend()
    
    # Plotting Epoch Times
    plt.subplot(1, 2, 2)
    plt.plot(epochs, epoch_times, label='Epoch Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Epoch Time Over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    sample_rate = 8000
    num_epochs  = 15

    # Create a new path by appending a relative path
    metadata_path = get_path('Metadata/metadata_small_8k.csv')
    
    main(sample_rate, metadata_path, num_epochs)
