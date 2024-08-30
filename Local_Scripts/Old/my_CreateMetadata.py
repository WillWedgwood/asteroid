import pandas as pd
import librosa
import os

def create_metadata(clean_paths, noise_paths, output_dir, snr_db):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    
    for clean_path in clean_paths:
        for noise_path in noise_paths:
            # Create a unique name for each mixture
            mixture_name = f"{os.path.basename(clean_path).split('.')[0]}_{os.path.basename(noise_path).split('.')[0]}"
            
            # Mix the audio and save to the output directory
            mixed_audio, sr = mix_audio(clean_path, noise_path, snr_db)
            mixture_path = os.path.join(output_dir, f"{mixture_name}.wav")
            librosa.output.write_wav(mixture_path, mixed_audio, sr)
            
            # Add metadata entry
            metadata.append({"mixture_path": mixture_path, "clean_path": clean_path, "noise_path": noise_path})
    
    # Save metadata to CSV file
    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_csv_path, index=False)
    
    return metadata_csv_path

clean_files = ["./librispeech_data/LibriSpeech/dev-clean/XXX/XXX.wav"]  # List of clean audio files
noise_files = ["./my_noise_dataset/YYY.wav"]  # List of noise audio files
output_directory = "./mixed_dataset"

metadata_csv = create_metadata(clean_files, noise_files, output_directory, snr_db)
