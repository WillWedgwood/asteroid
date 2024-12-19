import sys
from pathlib import Path
import os

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import torch
from asteroid.models import ConvTasNet, base_models
import soundfile as sf
from asteroid.utils.torch_utils import pad_x_to_y, jitable_shape
import numpy as np

def apply_inverse_mask(model, input_audio_path, output_dir, enable_mask_threshold, enable_smooth_mask, resample=True, force_overwrite=True):
    # Load pre-trained model
    model = ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k')

    # Separate audio
    waveform, sample_rate = sf.read(input_audio_path)
    waveform = torch.tensor(waveform).unsqueeze(0)  # Add batch dimension

    # Ensure the waveform tensor is of type float32
    waveform = waveform.to(torch.float32)

    # Remember shape to shape reconstruction, cast to Tensor for torchscript
    shape = jitable_shape(waveform)
    # Reshape to (batch, n_mix, time)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(1)

    # Real forward
    tf_rep = model.forward_encoder(waveform)
    est_masks = model.forward_masker(tf_rep)

    # Normalize est_masks to the range [0, 1]
    est_masks = est_masks / est_masks.max()

    # Debug: Print the shape and the maximum value of est_masks
    print(f"est_masks shape: {est_masks.shape}")
    print(f"est_masks max value: {est_masks.max().item()}")
    print(f"est_masks min value: {est_masks.min().item()}")

    if enable_smooth_mask:
        # Apply smoothing to the mask
        est_masks = smooth_mask(est_masks)

    # Apply inverse mask using in-place operation
    inverse_mask = torch.ones_like(est_masks) - est_masks

    if enable_mask_threshold:
        # Flatten the est_masks tensor for easier indexing
        flat_inverse_masks = inverse_mask.view(-1)

        # Apply threshold to the mask
        threshold = 0.99999  # Adjust this value based on your needs
        flat_inverse_masks = torch.where(flat_inverse_masks > threshold, flat_inverse_masks, torch.zeros_like(flat_inverse_masks))

        # Reshape the flat_est_masks back to the original shape
        inverse_mask = flat_inverse_masks.view(est_masks.shape)

    masked_tf_rep = model.apply_masks(tf_rep, inverse_mask) #est_masks)
    noise_decoded = model.forward_decoder(masked_tf_rep)

    noise = pad_x_to_y(noise_decoded, waveform)
    noise = base_models._shape_reconstructed(noise, shape)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Ensure noise is in the correct shape and type
    noise = noise.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to numpy
    if noise.ndim == 1:
        noise = noise[:, None]  # Add channel dimension if missing
    elif noise.shape[0] == 1:
        noise = noise.T  # Transpose to (samples, 1) if shape is (1, samples)
    noise = noise.astype('float32')  # Ensure type is float32

    # Normalize the audio
    noise = noise / np.max(np.abs(noise))  # Normalize the audio

    # Check for NaN or Inf values
    if np.any(np.isnan(noise)) or np.any(np.isinf(noise)):
        raise ValueError("The noise array contains NaN or Inf values.")

    # Save the noise audio
    output_noise_path = f"{output_dir}/noise_estimate.wav"
    sf.write(output_noise_path, noise, sample_rate)

    return output_noise_path

import torch.nn.functional as F

def smooth_mask(mask, kernel_size=5, sigma=1.0):
    # Create a Gaussian kernel
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    x = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel = x / x.sum()
    kernel = kernel.view(1, 1, -1).repeat(mask.size(1), 1, 1)
    
    # Apply the Gaussian filter
    mask = F.conv1d(mask, kernel, padding=kernel_size // 2, groups=mask.size(1))
    return mask


# Example usage
input_audio_path = "Datasets/Apple_Comms/ENGLISH_16K_Splits/52-COMMS MIX ENGLISH-240928_1934_23.wav"
output_dir = "Results/Apple_Comms/Post_Process"

enable_mask_threshold = True
enable_smooth_mask = False

apply_inverse_mask(ConvTasNet, input_audio_path, output_dir, enable_mask_threshold, enable_smooth_mask)