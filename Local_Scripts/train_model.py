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
from torch.optim.lr_scheduler import StepLR
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr, PairwiseNegSDR
import torch
import torchaudio
import csv
import numpy as np
import time  # Import time module for measuring elapsed time
from util import get_path
import matplotlib.pyplot as plt
import datetime

def check_sample_rate(file_path, expected_sample_rate):
    """Checks if the audio file has the expected sample rate."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != expected_sample_rate:
        raise ValueError(f"Sample rate mismatch in file {file_path}. "
                         f"Expected {expected_sample_rate}, but got {sample_rate}.")

def main(sample_rate, metadata_path):

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


    ### Test Set Up

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

    # Create the model
    model = ConvTasNet(n_src=1)

    # Define loss function and optimizer
    loss_fn = PairwiseNegSDR("sisdr")
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)


    ###### Define training, val and test loops ######

    # Define the training loop
    def train_epoch(model, train_loader, optimizer, loss_fn, device):
        model.train()
        total_loss = 0

        for batch in train_loader:
            mixtures, sources = batch
            mixtures, sources = mixtures.to(device), sources.to(device)

            optimizer.zero_grad()
            estimates = model(mixtures)
            loss = loss_fn(estimates, sources)
            loss = loss.mean() # we're trying this for now, since it returns a vector
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # print(f'Train Loss: {loss.item():.4f}')

        return total_loss / len(train_loader)

    # Define the validation loop
    def validate(model, val_loader, loss_fn, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                mixtures, sources = batch
                mixtures, sources = mixtures.to(device), sources.to(device)

                estimates = model(mixtures)
                loss = loss_fn(estimates, sources)
                loss = loss.mean()  # Reduce loss to a scalar value
                
                total_loss += loss.item()  # Accumulate loss
        return total_loss / len(val_loader)
    
    # Define the validation loop
    def test(model, train_loader, loss_fn, device):
        model.eval()
        total_loss = 0
        max_test_loss = 0
        
        with torch.no_grad():
            for batch in train_loader:
                mixtures, sources = batch
                mixtures, sources = mixtures.to(device), sources.to(device)

                estimates = model(mixtures)
                loss = loss_fn(estimates, sources)
                loss = loss.mean()  # Reduce loss to a scalar value

                total_loss += loss.item()  # Accumulate loss

                if abs(loss.item()) > abs(max_test_loss):
                    max_test_loss = loss.item()  # Update max loss if current loss is greater

        return total_loss / len(train_loader), max_test_loss


    ###### Training process ######
    
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    #device = 'cpu'
    print(f'Processing using: {device}')

    model.to(device)

    # Initialize arrays to store the loss values and epoch times
    train_losses = []
    val_losses = []
    epoch_times = []

    # Create a report file with a timestamp
    report_filename = f"Trained_Models/Training_Reports/training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_filename, "w") as report_file:

        num_epochs = 50  # low for testing purposes
        for epoch in range(num_epochs):

            # Record the start time for the epoch
            epoch_start_time = time.time()

            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss = validate(model, val_loader, loss_fn, device)
            scheduler.step()  # Update learning rate

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
    avg_test_loss, max_test_loss = test(model, test_loader, loss_fn, device)

    # Optionally, save the model
    torch.save(model.state_dict(), 'model_bankhol.pth')

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

    # Create a new path by appending a relative path
    metadata_path = get_path('Metadata/metadata_dev_8k.csv')
    
    main(sample_rate, metadata_path)









#### Tried to write code the save all the output wavs but it wasn't really working well.

    # # Function to save estimates as WAV files
    # def save_estimates_as_wav(estimates, output_dir, sample_rate=16000):
    #     """Saves model estimates as mono WAV files in the specified directory."""
    #     os.makedirs(output_dir, exist_ok=True)
    #     saved_files = []
        
    #     for i, estimate in enumerate(estimates):
    #         # Convert tensor to numpy array and ensure correct shape
    #         estimate = estimate.squeeze().cpu().numpy()  # Squeeze to remove extra dimensions
    #         if estimate.ndim == 1:
    #             estimate = estimate[np.newaxis, :]  # Add a channel dimension if needed
            
    #         # Generate filename and save WAV file
    #         file_path = os.path.join(output_dir, f"{i + 1}.wav")
    #         try:
    #             torchaudio.save(file_path, torch.tensor(estimate).unsqueeze(0), sample_rate)
    #             saved_files.append(file_path)
    #             print(f"Saved {file_path}")
    #         except Exception as e:
    #             print(f"Failed to save {file_path}: {e}")
    #             raise
        
    #     return saved_files


    # # Define the testing loop
    # def test(model, test_loader, device, test_result_path, sample_rate=16000):
    #     """Tests the model and saves estimates as mono WAV files."""
    #     model.eval()
    #     results = []
        
    #     with torch.no_grad():
    #         for batch in test_loader:
    #             mixtures, sources = batch
    #             mixtures, sources = mixtures.to(device), sources.to(device)

    #             # Generate estimates
    #             estimates = model(mixtures)
                
    #             # Save each estimate as a WAV file
    #             saved_files = save_estimates_as_wav(estimates, test_result_path, sample_rate)
                
    #             # Get input filenames from dataset
    #             input_filenames = [os.path.basename(file) for file in test_loader.dataset.df['mixture_path'].tolist()]
                
    #             # Log results
    #             for input_filename, saved_file in zip(input_filenames, saved_files):
    #                 results.append((input_filename, os.path.basename(saved_file)))
        
    #     # Save the test results to a CSV file
    #     results_csv_path = os.path.join(test_result_path, 'test_results.csv')
    #     with open(results_csv_path, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['input_file', 'output_file'])
    #         writer.writerows(results)
        
    #     return results