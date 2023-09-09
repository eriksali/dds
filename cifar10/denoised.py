import torch

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the number of samples to generate
num_samples = 10

# Define the latent space dimension (size of random noise)
latent_dim = 100

# Create random noise samples
noise = torch.randn(num_samples, latent_dim).to(device)

# Generate samples using the trained generator
with torch.no_grad():
    generated_samples = generator(noise)

# You can now save, display, or further process the generated samples
# For example, if you want to save the generated images as image files:
from torchvision.utils import save_image

# Define a directory to save the generated samples
output_dir = "generated_samples/"

# Ensure the output directory exists
import os
os.makedirs(output_dir, exist_ok=True)

# Save each generated sample as an image file
for i, sample in enumerate(generated_samples):
    save_image(sample, os.path.join(output_dir, f"sample_{i}.png"))
