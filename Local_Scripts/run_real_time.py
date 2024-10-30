import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import sys
from pathlib import Path
import time
import torch
from asteroid.models import ConvTasNet
import torchaudio
import numpy as np
from util import get_path

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
    hop_length = int(window_length * (1 - overlap))  # Effective hop size between chunks

    num_samples = waveform.size(1)
    num_channels = waveform.size(0)

    # Initialize output buffer and overlap tracking
    output_buffer = torch.zeros(num_channels, num_samples)
    overlap_buffer = torch.zeros(num_channels, window_length)
    
    total_time = 0
    position = 0

    with torch.no_grad():
        # Process each chunk
        for start in range(0, num_samples, hop_length):
            end = min(start + window_length, num_samples)
            chunk = waveform[:, start:end]
            
            if chunk.size(1) < window_length:
                # Zero-pad the last chunk if it's shorter than the window length
                padding = window_length - chunk.size(1)
                chunk = torch.nn.functional.pad(chunk, (0, padding))

            # Add batch dimension: [1, num_channels, num_samples]
            chunk = chunk.unsqueeze(0)

            # Measure processing time for this chunk
            start_time = time.time()
            estimate = model(chunk)
            processing_time = time.time() - start_time
            total_time += processing_time
            
            estimate = estimate.squeeze(0)  # Remove batch dimension
            
            # Add the processed chunk to the output buffer using overlap-add method
            overlap_length = output_buffer[:, position:position + window_length].size(1)
            output_buffer[:, position:position + window_length] += estimate[:, :overlap_length]
            position += hop_length
    
    return output_buffer[:, :num_samples].cpu().numpy(), total_time


def save_audio(estimate, output_path, sample_rate):
    """Save the processed estimate as an audio file (WAV format)."""
    estimate = torch.tensor(estimate)
    
    # Ensure estimate is 2D (num_channels, num_samples)
    if estimate.ndimension() == 1:
        estimate = estimate.unsqueeze(0)  # Add channel dimension if it's mono

    # Convert tensor to a 2D NumPy array
    estimate_np = estimate.numpy()

    # Save the audio file
    torchaudio.save(output_path, torch.tensor(estimate_np), sample_rate)

def normalize_audio(audio):
    # Normalize audio to range [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / (max_val * 2) 
    return audio

def main(input_audio_path, output_audio_path, sample_rate, window_length_ms=100, overlap=0.5):
    
    # Load the pre-trained ConvTasNet model
    model = ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k')
    #model = ConvTasNet.from_pretrained('mpariente/ConvTasNet_WHAM_sepclean_8k')
    #model = ConvTasNet.from_pretrained('asteroid/ConvTasNet_WHAMR_sepclean_16k')
    #model = ConvTasNet.from_pretrained('mpariente/ConvTasNet_Libri1Mix_enhsingle_8k')

    # Load the input audio file
    waveform, sr = torchaudio.load(input_audio_path)
    
    if sr != sample_rate:
        raise ValueError(f"Sample rate of {input_audio_path} does not match the model's sample rate of {sample_rate}.")
    
    # Ensure waveform is 2D (num_channels, num_samples)
    if waveform.ndim == 1:
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
    input_audio_path = get_path('Datasets/Apple_Comms/SPANISH_16k_SPLITS/23-COMMENTARY SPANISH 2-240928_1934_23.wav')  # Path to the input audio file
    output_audio_path = get_path('Results/Apple_Comms/Real-Time/Spanish_JorisCos_16k_100ms.wav')  # Path to save the output WAV file

    sample_rate = 16000
    window_length_ms = 100  # 100 ms window length (adjustable)
    overlap = 0.5  # 50% overlap between windows (adjustable)

    main(input_audio_path, output_audio_path, sample_rate, window_length_ms, overlap)