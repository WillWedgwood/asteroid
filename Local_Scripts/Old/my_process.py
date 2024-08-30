import os
import torch
from asteroid.models import ConvTasNet
import soundfile as sf
from asteroid import separate
from asteroid.separate import _load_audio, _resample, torch_separate
import numpy as np
from asteroid.utils import get_device
import librosa

# Load the pre-trained ConvTasNet model
model = ConvTasNet.from_pretrained('mhu-coder/ConvTasNet_Libri1Mix_enhsingle')

def process_file(input_audio_path, output_directory):

    # Perform separation
    # separate.separate(model, input_audio_path, output_directory, resample=True, force_overwrite=True, segmented_audio=True)
    # print(f'Finished speech separation on {os.path.basename(input_audio_path)}')

    segment_audio = True #todo: pass this to the process function

    # So I've brought the processing from separate into this script, it does the segmenting on the array and then adds the extra dimension for batch size etc
    # Not sure why the change in shape has to happen. Currently processing takes way too long.

    # Estimates will be saved as filename_est1.wav etc...
    base, _ = os.path.splitext(input_audio_path)
    if output_directory is not None:
        base = os.path.join(output_directory, os.path.basename(base))
    save_name_template = base + "_est{}.wav"

    # SoundFile wav shape: [time, n_chan]
    wav, fs = _load_audio(input_audio_path)

   #ß wav = _resample(wav, fs, 16000)
    wav = _resample(wav[:, 0], orig_sr=fs, target_sr=int(model.sample_rate))[:, None]

    if segment_audio:
        frame_size = 4096
        total_frames = (len(wav) + frame_size - 1) // frame_size
        separated_frames = []

        for i in range(total_frames):
            start = i * frame_size
            end = min((i + 1) * frame_size, len(wav))
            frame = wav[start:end]

            if len(frame) < frame_size:
                padding = np.zeros(frame_size - len(frame))
                frame = np.concatenate([frame, padding])

            # Pass wav as [batch, n_chan, time]; here: [1, chan, time]
            # Why do we need to do this?
            frame = frame.T[None]

            ### Numpy Separate ###
            torch_frame = torch.from_numpy(frame)

            ### Torch Separate ###
            # Handle device placement
            input_device = get_device(torch_frame, default="cpu")
            model_device = get_device(model, default="cpu")
            torch_frame = torch_frame.to(model_device)
            # Forward
            separate_func = getattr(model, "forward_wav", model)
            out_frames = separate_func(torch_frame)

            # FIXME: for now this is the best we can do.
            out_frames *= torch_frame.abs().sum() / (out_frames.abs().sum())

            out_frames = out_frames.data.numpy()
            separated_frames.append(out_frames)

            if len(separated_frames) == 300:
                break

        separated_audio = []

        for i in range(len(separated_frames[0])):
            channel_audio = np.concatenate([frame[0, i] for frame in separated_frames], axis=0)
            separated_audio.append(channel_audio)
           # separated_audio = _resample(separated_audio, orig_sr=int(model.sample_rate), target_sr=fs)
        
        separated_audio = np.asarray(separated_audio, dtype=np.float32).flatten()
        #sf.write("test_output.wav", separated_audio, 16000)
        sf.write(save_name_template.format(1), separated_audio, 16000)

    else:
        # Pass wav as [batch, n_chan, time]; here: [1, chan, time]
        wav = wav.T[None]

        (est_srcs,) = numpy_separate(model, wav, **kwargs)
        # Resample to original sr
        est_srcs = [
            _resample(est_src, orig_sr=int(model.sample_rate), target_sr=fs) for est_src in est_srcs
        ]

        # Save wav files to filename_est1.wav etc...
        for src_idx, est_src in enumerate(est_srcs, 1):
            sf.write(save_name_template.format(src_idx), est_src, fs)

def process_directory(input_directory, base_output_directory):
    # Extract the name of the input directory
    input_dir_name = os.path.basename(os.path.normpath(input_directory))
    
    # Create the specific output directory under base output directory
    output_directory = os.path.join(base_output_directory, input_dir_name)
    os.makedirs(output_directory, exist_ok=True)
    
    # Iterate over all files in the input directory
    for file_name in os.listdir(input_directory):
        # Check if the file is a wav file
        if file_name.endswith('.wav'):
            input_audio_path = os.path.join(input_directory, file_name)
            # Perform separation
            process_file(input_audio_path, output_directory)

def main(input_path):
    base_output_directory = "Results"
    
    if os.path.isdir(input_path):
        print(f"Directory detected: Processing all WAV files in '{input_path}'")
        process_directory(input_path, base_output_directory)
    elif os.path.isfile(input_path) and input_path.endswith('.wav'):
        print(f"Single WAV file detected: Processing '{input_path}'")
        process_file(input_path, base_output_directory)
    else:
        print(f"Invalid input: {input_path}. Please provide a directory or a single WAV file.")

# Example usage
input_path = "Audio_Sources/Sky/Football/Football_Left_splits/Football_Left_4.wav"  # Replace with your input directory or file path
main(input_path)