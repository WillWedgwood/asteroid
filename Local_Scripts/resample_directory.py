import os
import torchaudio
from util import get_path

def resample_audio_files(input_folder, target_sample_rate):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"The folder {input_folder} does not exist.")
        return

    # Create the output folder name by adding '_{target_sample_rate}Hz' to the input folder's name
    output_folder = input_folder.parent.parent / f"{input_folder.name}_{target_sample_rate}Hz"
    output_folder.mkdir(exist_ok=True)
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith((".flac", ".wav")):
                # Construct the full file path
                input_path = os.path.join(root, filename)
                
                # Load the audio file
                waveform, sample_rate = torchaudio.load(input_path)
                
                # Check if resampling is needed
                if sample_rate != target_sample_rate:
                    # Resample the audio
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                    waveform = resampler(waveform)
                
                # Define the output file path (keeping the original extension)
                output_path = os.path.join(output_folder, filename)
                
                # Ensure the output filename is unique if there are duplicates
                base_name, ext = os.path.splitext(output_path)
                counter = 1
                while os.path.exists(output_path):
                    output_path = f"{base_name}_{counter}{ext}"
                    counter += 1
                
                # Save the resampled audio to the output folder
                torchaudio.save(output_path, waveform, target_sample_rate)
                
                print(f"Resampled {filename} to {target_sample_rate}Hz and saved to {output_folder}")

# Example usage:
input_folder = get_path('Datasets/Clean_Datasets/LibriSpeech_Datasets/LibriSpeech/train-clean-100')  # Replace with your folder path
target_sr = 8000

resample_audio_files(input_folder, target_sr)
