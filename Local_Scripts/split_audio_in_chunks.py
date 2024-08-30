import os
import soundfile as sf
from util import get_path

def split_audio(file_path):
    # Load the audio file
    audio, sample_rate = sf.read(file_path)

    # Calculate the number of samples per segment (1 minute)
    segment_samples = sample_rate * 60

    # Get the file name and directory
    file_dir, file_name = os.path.split(file_path)
    file_base, file_ext = os.path.splitext(file_name)

    # Create the output directory
    output_dir = os.path.join(file_dir, f"{file_base}_splits")
    os.makedirs(output_dir, exist_ok=True)

    # Split the audio file into segments
    total_samples = len(audio)
    num_segments = (total_samples + segment_samples - 1) // segment_samples  # Ceiling division

    for i in range(num_segments):
        start_sample = i * segment_samples
        end_sample = min(start_sample + segment_samples, total_samples)
        segment = audio[start_sample:end_sample]

        # Create the output file name
        output_file = os.path.join(output_dir, f"{file_base}_{i + 1}.wav")
        
        # Export the segment
        sf.write(output_file, segment, sample_rate)
        print(f"Exported {output_file}")

# Example usage
split_audio(get_path("Audio_Sources/Sky/Football/Football_Right.wav"))
