import torch
from asteroid.models import ConvTasNet, FasNetTAC, DeMask
import soundfile as sf
from asteroid import separate

### Load the pre-trained ConvTasNet model ###

# model = ConvTasNet.from_pretrained('mpariente/ConvTasNet_WHAM!_sepclean')
# model = ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k')
model = ConvTasNet.from_pretrained('mhu-coder/ConvTasNet_Libri1Mix_enhsingle')

# Load the FasNet model -- doesn't work!
# model = FasNetTAC.from_pretrained('popcornell/FasNetTAC_TACDataset_separatenoisy')

# Load the DeMask model
# model = DeMask.from_pretrained('popcornell/DeMask_Surgical_mask_speech_enhancement_v1')

# # Path to your input audio file
# input_audio_path = "Audio_Sources/wolves_commentary_mono_short.wav" # "Audio_Sources/MCvsCHE_16k.wav" 
# separate.separate(model, input_audio_path, 'Results', resample=False)

# STEREO TEST
input_audio_path = "Audio_Sources/Sky/Football/Football_Right_splits/Football_Right_2.wav" # "Audio_Sources/MCvsCHE_16k.wav" 
separate.separate(model, input_audio_path, 'Results', resample=True, force_overwrite=True)

# # Calculate Noise
# input_wav, fs = sf.read(input_audio_path)
# speech, fs = sf.read('/Results/MCvsCHE_16k_est1.wav')

# noise = input_wav - speech

# sf.write('Results/Noise.wav', noise)