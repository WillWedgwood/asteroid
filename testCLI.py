import os
from asteroid.models import ConvTasNet

# Step 1: Pull the ConvTasNet model from asteroid.models
model = ConvTasNet.from_pretrained('mpariente/ConvTasNet_WHAM!_sepclean')

# Step 2: Define the parameters for the run.sh script
stage = 3
tag = "wills_test"
task = "sep_noisy"
id = "0,1"

# Define the path to the run.sh script
script_path = "egs/wham/ConvTasNet/run.sh"

# Create the command to execute the script with the specified parameters
command = f"./{script_path} --stage {stage} --tag {tag} --task {task} --id {id}"

# Step 3: Execute the run.sh script
os.system(command)