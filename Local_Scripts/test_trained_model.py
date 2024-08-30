import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import torch
import torchaudio
import numpy as np
from asteroid.models import ConvTasNet
from util import get_path

## TODO: Process in batches?

def load_model(model_path):
    """Load the pre-trained model from a .pth file."""
    model = ConvTasNet(n_src=1)  # Adjust `n_src` if necessary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

def process_audio(model, audio_path, sample_rate):
    """Process an audio file (WAV or FLAC) with the loaded model."""
    waveform, sr = torchaudio.load(audio_path)
    
    if sr != sample_rate:
        raise ValueError(f"Sample rate of {audio_path} does not match the model's sample rate of {sample_rate}.")
    
    # Ensure waveform is 2D (num_channels, num_samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension if it's a mono file
    
    with torch.no_grad():
        # Add batch dimension and pass through the model
        waveform = waveform.unsqueeze(0)  # Now waveform is [batch_size, num_channels, num_samples]
        estimate = model(waveform)
    
    return estimate.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU

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

def main(model_path, input_audio_path, output_audio_path, sample_rate):
    # Load the pre-trained model
    model = load_model(model_path)

    # Process the input audio file
    estimate = process_audio(model, input_audio_path, sample_rate)

    # Save the processed output
    save_audio(estimate, output_audio_path, sample_rate)
    print(f"Processed audio file saved at: {output_audio_path}")

if __name__ == '__main__':
    model_path = get_path('Trained_Models/model_bankhol.pth')  # Path to your .pth model file
    input_audio_path = get_path('Datasets/Real_Commentary/Sky/Football/Football_Left_splits_8000Hz/Football_Left_2.wav')  # Path to the input audio file
    output_audio_path = get_path('Results/output_bankhol.wav')  # Path to save the output WAV file

    sample_rate = 8000

    main(model_path, input_audio_path, output_audio_path, sample_rate)
