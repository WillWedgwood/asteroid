import sys
from pathlib import Path
import os

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import torch
from asteroid.models import ConvTasNet

# Load pre-trained model
model = ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k')
model.eval()

# Dummy input for tracing
dummy_input = torch.randn(1, 1, 2048)  # Adjust the shape based on your model's expected input

# Export the model to ONNX with a specific opset version
torch.onnx.export(model, dummy_input, "conv_tasnet_JorisCos_2048.onnx", opset_version=11)