import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from util import get_path

def apply_fade_window(segment, fade_length):
    """Apply a fade-in and fade-out window to the segment."""
    window = np.hanning(2 * fade_length)
    fade_in = window[:fade_length]
    fade_out = window[fade_length:]
    
    segment[:fade_length] *= fade_in
    segment[-fade_length:] *= fade_out
    
    return segment

def split_into_chunks(audio, chunk_length, sample_rate):
    """Split audio into chunks of specified length, ignoring the last segment if smaller than chunk_length."""
    num_samples = len(audio)
    chunk_size = int(chunk_length * sample_rate)
    chunks = [audio[i:i + chunk_size] for i in range(0, num_samples - chunk_size + 1, chunk_size)]
    return chunks

def mix_audio(clean_audio, noise_audio, snr_range):
    """Mix clean audio with noise at a random SNR within the specified range."""
    snr = np.random.uniform(*snr_range)
    snr_linear = 10 ** (snr / 20.0)

    clean_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    scaling_factor = np.sqrt(clean_power / (noise_power * snr_linear))

    # Scale the noise and mix with clean audio
    mixed_audio = clean_audio + scaling_factor * noise_audio

    # Ensure the output is in the valid range [-1, 1]
    mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))

    return mixed_audio, snr

def extract_speech_and_noise_segments(input_speech_path, input_noise_path, output_dir, snr_range, chunk_length=3, fade_length=100, sample_rate=16000):
    # Load the VAD model
    model = load_silero_vad()

    # Load the input speech and noise WAV files
    speech_wav, speech_sample_rate = librosa.load(input_speech_path, sr=None)
    noise_wav, noise_sample_rate = librosa.load(input_noise_path, sr=None)

    # Resample to 16kHz if necessary
    if speech_sample_rate != sample_rate:
        speech_wav = librosa.resample(speech_wav, orig_sr=speech_sample_rate, target_sr=sample_rate)
        speech_sample_rate = sample_rate

    if noise_sample_rate != sample_rate:
        noise_wav = librosa.resample(noise_wav, orig_sr=noise_sample_rate, target_sr=sample_rate)
        noise_sample_rate = sample_rate

    # Ensure both files have the same sample rate
    if speech_sample_rate != noise_sample_rate:
        raise ValueError("Sample rates of speech and noise files do not match.")

    # Ensure both files are the same length
    min_length = min(len(speech_wav), len(noise_wav))
    speech_wav = speech_wav[:min_length]
    noise_wav = noise_wav[:min_length]

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(speech_wav, model, return_seconds=False)

    # Extract speech and corresponding noise segments
    speech_segments = []
    noise_segments = []
    for timestamp in speech_timestamps:
        start = timestamp['start']
        end = timestamp['end']
        speech_segment = speech_wav[start:end]
        noise_segment = noise_wav[start:end]

        # Apply fade-in and fade-out window
        speech_segment = apply_fade_window(speech_segment, fade_length)
        noise_segment = apply_fade_window(noise_segment, fade_length)

        speech_segments.append(speech_segment)
        noise_segments.append(noise_segment)

    # Concatenate speech and noise segments into new arrays
    speech_array = np.concatenate(speech_segments)
    noise_array = np.concatenate(noise_segments)

    # Split into 3-second chunks, ignoring the last segment if smaller than chunk_length
    speech_chunks = split_into_chunks(speech_array, chunk_length, sample_rate)
    noise_chunks = split_into_chunks(noise_array, chunk_length, sample_rate)

    # Extract sport name from input paths
    sport_name = Path(input_speech_path).parts[-2]

    # Create sport subdirectory within the output directory
    sport_dir = os.path.join(output_dir, sport_name)
    os.makedirs(sport_dir, exist_ok=True)

    # Create "clean" and "mixed" folders within the sport subdirectory
    clean_dir = os.path.join(sport_dir, 'clean')
    mixed_dir = os.path.join(sport_dir, 'mixed')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(mixed_dir, exist_ok=True)

    metadata = []

    # Mix chunks and save
    for i, (speech_chunk, noise_chunk) in enumerate(zip(speech_chunks, noise_chunks)):
        # Apply window function to each chunk
        speech_chunk = apply_fade_window(speech_chunk, fade_length)
        noise_chunk = apply_fade_window(noise_chunk, fade_length)

        mixed_audio, snr = mix_audio(speech_chunk, noise_chunk, snr_range)

        # Save the mixed audio
        mixed_output_filename = f"chunk_{i}_snr{snr:.1f}.wav"
        mixed_output_path = os.path.join(mixed_dir, mixed_output_filename)
        sf.write(mixed_output_path, mixed_audio, sample_rate)

        # Save the clean speech reference
        clean_output_filename = f"chunk_{i}_clean.wav"
        clean_output_path = os.path.join(clean_dir, clean_output_filename)
        sf.write(clean_output_path, speech_chunk, sample_rate)

        # Append metadata
        metadata.append({
            'mixture_path': mixed_output_path,
            'source_1_path': clean_output_path,
            'snr': snr,
            'length': len(speech_chunk)  # Length of chunk in samples
        })

    return metadata

def process_multiple_folders(parent_dir, output_dir, snr_range, chunk_length=3, fade_length=100, sample_rate=16000):
    all_metadata = []

    # Find all subdirectories within the parent directory
    input_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    for input_dir in input_dirs:
        input_speech_path = os.path.join(input_dir, 'PGM_Centre.wav')
        input_noise_path = os.path.join(input_dir, 'PGM_Left.wav')

        if os.path.exists(input_speech_path) and os.path.exists(input_noise_path):
            metadata = extract_speech_and_noise_segments(input_speech_path, input_noise_path, output_dir, snr_range, chunk_length, fade_length, sample_rate)
            all_metadata.extend(metadata)
        else:
            print(f"Skipping {input_dir} as it does not contain both PGM_Centre.wav and PGM_Left.wav")

    # Save all metadata to a single CSV file
    metadata_df = pd.DataFrame(all_metadata)
    metadata_csv_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_csv_path, index=False)

    print(f"All metadata saved at: {metadata_csv_path}")

if __name__ == '__main__':
    parent_dir = 'Datasets/NBC_Data/NBC_Data_16000Hz_checked'  # Parent directory containing subdirectories with PGM_Centre.wav and PGM_Left.wav
    output_dir = 'Datasets/NBC_Data/NBC_Training_Data_Dec_2024'  # Path to save the extracted speech and noise chunks
    snr_range = (5, 30)  # Desired SNR range
    chunk_length = 3  # Chunk length in seconds
    fade_length = 100  # Fade length in samples
    sample_rate = 16000  # Sample rate

    process_multiple_folders(parent_dir, output_dir, snr_range, chunk_length, fade_length, sample_rate)