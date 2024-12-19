import sys
from pathlib import Path
import os

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import os
import torch
import torchaudio
import numpy as np
import time
from asteroid.models import ConvTasNet

def apply_windowing(chunk, window):
    return chunk * window

def process_audio_chunked(model, waveform, sample_rate, window_length_ms=100, overlap=0.5):
    """
    Process an audio signal in chunks with a specified window length and overlap.
    
    Parameters:
    - model: Pre-trained model to process audio.
    - waveform: Input waveform to be processed.
    - sample_rate: Sampling rate of the audio.
    - window_length_ms: Window length in milliseconds.
    - overlap: Overlap percentage between consecutive windows (0 to 1).
    
    Returns:
    - estimate_full: Processed audio signal after processing each chunk.
    - total_time: Total processing time.
    """
    # Convert window length from ms to samples
    window_length = int((window_length_ms / 1000) * sample_rate)
    
    # Calculate hop length based on overlap
    hop_length = int(window_length * (1 - overlap))  # Effective hop size between chunks

    num_samples = waveform.size(1)
    num_channels = waveform.size(0)

    # Initialize output buffer and overlap tracking
    output_buffer = torch.zeros(num_channels, num_samples)
    overlap_buffer = torch.zeros(num_channels, window_length)
    
    # Create the windowing function (Hamming window)
    window = torch.hamming_window(window_length, periodic=False).to(waveform.device)

    total_time = 0
    position = 0

    with torch.no_grad():
        # Process each chunk with overlap
        for start in range(0, num_samples, hop_length):
            end = min(start + window_length, num_samples)
            chunk = waveform[:, start:end]
            
            if chunk.size(1) < window_length:
                # Zero-pad the last chunk if it's shorter than the window length
                padding = window_length - chunk.size(1)
                chunk = torch.nn.functional.pad(chunk, (0, padding))

            # Apply the windowing function
            chunk = apply_windowing(chunk, window)

            # Add batch dimension: [1, num_channels, num_samples]
            chunk = chunk.unsqueeze(0)

            # Measure processing time for this chunk
            start_time = time.time()
            estimate = model(chunk)
            processing_time = time.time() - start_time
            total_time += processing_time

            # Remove batch dimension and overlap-add
            estimate = estimate.squeeze(0)
            output_buffer[:, start:end] += estimate[:, :end-start]

    return output_buffer[:, :num_samples].cpu().numpy(), total_time

def save_audio(estimate, output_path, sample_rate):
    """Save the processed estimate as an audio file (WAV format)."""
    estimate = torch.tensor(estimate)
    
    # Ensure estimate is 2D (num_channels, num_samples)
    if estimate.ndimension() == 1:
        estimate = estimate.unsqueeze(0)  # Add channel dimension if it's mono

    # Convert tensor to a 2D NumPy array
    estimate_np = estimate.numpy()

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the audio file
    torchaudio.save(output_path, torch.tensor(estimate_np), sample_rate)

def normalize_audio(audio):
    # Normalize audio to range [-1, 1]
    max_val = np.max(np.abs(audio))
    if (max_val > 0):
        audio = audio / max_val
    return audio

def main(input_audio_path, output_audio_path, sample_rate, window_length_ms=100, overlap=0.5):
    # Load the pre-trained ConvTasNet model
    model = ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k')

    # Load the input audio file
    waveform, sr = torchaudio.load(input_audio_path)
    
    if sr != sample_rate:
        raise ValueError(f"Sample rate of {input_audio_path} does not match the model's sample rate of {sample_rate}.")

    # Ensure waveform is 2D (num_channels, num_samples)
    if waveform.ndimension() == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension if it's mono
    
    # Process the audio file in chunks
    estimate, total_time = process_audio_chunked(model, waveform, sample_rate, window_length_ms, overlap)

    # Normalize the processed output
    normalized_estimate = normalize_audio(estimate)

    # Save the processed and normalized output
    save_audio(normalized_estimate, output_audio_path, sample_rate)
    
    # Calculate total duration of the audio
    duration = waveform.size(1) / sample_rate

    print(f"Processed audio file saved at: {output_audio_path}")
    print(f"Total audio duration: {duration:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Processing speed: {duration / total_time:.2f}x real-time")

if __name__ == '__main__':
    input_audio_path = 'Datasets/Apple_Comms/SPANISH_16k_SPLITS/23-COMMENTARY SPANISH 2-240928_1934_15.wav '  # Path to the input audio file
    output_audio_path = 'Results/Apple_Comms/Real-Time/Spanish_JorisCos_16k_128ms.wav'  # Path to save the output WAV file

    sample_rate = 16000
    window_length_ms = 128  # 200 ms window length (adjustable)
    overlap = 0.5  # 70% overlap between windows (adjustable)

    main(input_audio_path, output_audio_path, sample_rate, window_length_ms, overlap)