import torch
from asteroid.models import ConvTasNet, DeMask
import soundfile as sf
from asteroid import separate
import os

# Define the list of models to test
models = [
    ConvTasNet.from_pretrained("popcornell/DPRNNTasNet_WHAM_enhancesingle"),
    ConvTasNet.from_pretrained("brijmohan/ConvTasNet_Libri1Mix_enhsingle"),
    ConvTasNet.from_pretrained("mhu-coder/ConvTasNet_Libri1Mix_enhsingle"),  # THIS IS THE BEST
    DeMask.from_pretrained('popcornell/DeMask_Surgical_mask_speech_enhancement_v1')
]

# Define the input audio path
input_audio_path = "Audio_Sources/Sky/Football/Football_Left_splits/Football_Left_2.wav"

# Loop through the models and save the results
for i, model in enumerate(models, start=1):
    output_dir = f"Enh_Results/Model{i}/"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    separate.separate(model, input_audio_path, output_dir, resample=True, force_overwrite=True)

    print(f"Processed model {i}, results saved in {output_dir}")
