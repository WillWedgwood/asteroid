import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from glob import glob
from util import get_path

def get_audio_files(directory):
    """Recursively get all .wav and .flac audio files in the directory."""
    return [y for x in os.walk(directory) for y in glob(os.path.join(x[0], '*.wav')) + glob(os.path.join(x[0], '*.flac'))]


def mix_audio_with_noise(clean_dir, noise_dir, output_dir, snr_range, sample_rate=16000):
    """
    Mix clean audio files with noise at random SNRs within a specified range and save the mixed files.

    Args:
        clean_dir (str): Path to the directory containing clean audio files.
        noise_dir (str): Path to the directory containing noise audio files.
        output_dir (str): Path to the directory to save the mixed audio files.
        snr_range (tuple): Range of SNRs to use for mixing (e.g., (10, 30)).
        sample_rate (int): Sample rate for the audio files.

    Returns:
        metadata (pd.DataFrame): DataFrame containing metadata for the mixed audio files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Recursively get list of clean and noise files
    clean_files = get_audio_files(clean_dir)
    noise_files = get_audio_files(noise_dir)

    metadata = []

    # Load the first noise file
    noise_idx = 0
    noise_audio, _ = librosa.load(noise_files[noise_idx], sr=sample_rate)
    noise_pointer = 0

    for clean_file in clean_files:
        # Load clean audio
        clean_audio, _ = librosa.load(clean_file, sr=sample_rate)

        # Get length of clean audio in samples
        clean_length = len(clean_audio) #/ sample_rate

        # Determine the required length of noise
        noise_required_len = len(clean_audio)

        # Accumulate noise from the current noise file
        if noise_pointer + noise_required_len > len(noise_audio):
            # If the current noise file is insufficient, switch to the next one
            noise_idx = (noise_idx + 1) % len(noise_files)
            new_noise_audio, _ = librosa.load(noise_files[noise_idx], sr=sample_rate)
            noise_audio = np.concatenate((noise_audio[noise_pointer:], new_noise_audio))
            noise_pointer = 0

        # Select the segment of noise
        noise_segment = noise_audio[noise_pointer:noise_pointer + noise_required_len]
        noise_pointer += noise_required_len

                # Apply a random SNR for each clean sample
        snr = np.random.uniform(*snr_range)
        snr_linear = 10 ** (snr / 20.0)

        clean_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise_segment ** 2)
        scaling_factor = np.sqrt(clean_power / (noise_power * snr_linear))

        # Scale the noise and mix with clean audio
        mixed_audio = clean_audio + scaling_factor * noise_segment

        # Ensure the output is in the valid range [-1, 1]
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))

        # Save the mixed audio
        output_filename = f"{os.path.splitext(os.path.basename(clean_file))[0]}_snr{snr:.1f}.wav"
        output_path = os.path.join(output_dir, output_filename)
        sf.write(output_path, mixed_audio, sample_rate)

        # # Save the noise audio used
        # noise_output_filename = f"{os.path.splitext(os.path.basename(clean_file))[0]}_noise_snr{snr:.1f}.wav"
        # noise_output_path = os.path.join(output_dir, noise_output_filename)
        # sf.write(noise_output_path, scaling_factor * noise_segment, sample_rate)

        # Append metadata
        metadata.append({
            'mixture_path': output_path,
            'source_1_path': clean_file,
            # 'source_2_path': noise_output_path,
            # 'snr': snr,
            'length': clean_length  # Add length of clean audio in seconds
        })

    # Save metadata to CSV file
    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = get_path('Metadata/metadata_dev_8k.csv')
    metadata_df.to_csv(metadata_csv_path, index=False)

    return metadata_csv_path

clean_dir = get_path('Datasets/Clean_Datasets/LibriSpeech_Datasets/dev-clean_8000Hz')  # Clean Speech Dir   "LibriSpeech_Small"
noise_dir = get_path("Datasets/Noise_Datasets/Noise_Dev")  # Noise Dir
output_dir = get_path("Datasets/Mixed_Datasets/Mix_Output_Dev_8k")
snr_range = (5, 20)  # Desired SNR in 
sample_rate = 8000

mix_audio_with_noise(clean_dir, noise_dir, output_dir, snr_range, sample_rate)
