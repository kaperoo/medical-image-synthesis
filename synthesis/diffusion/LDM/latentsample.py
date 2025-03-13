import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms, utils
import numpy as np
from latentmodel import LatentConditionalUnet
from latenttraining import linear_beta_schedule, get_index_from_list
from autoencoder import Autoencoder  # Import the pretrained autoencoder
import os

SAVE_PATH = "."
MODEL_NAME = "latentmodel.pth"
AUTOENCODER_PATH = "autoencoder.pth"  # Path to the trained autoencoder
# IMG_SIZE = (128, 288)  # Image resolution
IMG_SIZE = (128, 352)  # Image resolution
# IMG_SIZE = (208, 560)  # Image resolution
LATENT_DIM = 4  # Must match the encoder
DOWNSAMPLE_FACTOR = 4  # Encoder downsampling factor
T = 2000  # Number of diffusion timesteps
TAG = "LDM"

def show_tensor_image(image):
    """
    Converts a PyTorch tensor to a PIL image, ensuring proper scaling.
    """
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),  # Scale to [0, 255]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  # Convert to uint8
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    
    return reverse_transforms(image)

# Define beta schedule
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed-form solutions
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

@torch.no_grad()
def sample_timestep(z, class_label, t):
    """
    Performs a reverse diffusion step in **latent space**.
    """
    betas_t = get_index_from_list(betas, t, z.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, z.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, z.shape)

    # Predict noise in latent space
    model_mean = sqrt_recip_alphas_t * (
        z - betas_t * model(z, t, class_label) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, z.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(z)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def generate_latent_images(n=7, tag='ldm'):
    """
    Generate images using the **Latent Diffusion Model (LDM)**.
    """
    save_dir = f"generated/{tag}"
    joined = os.path.join(SAVE_PATH, save_dir)
    os.makedirs(joined, exist_ok=True)

    latent_size = (IMG_SIZE[0] // DOWNSAMPLE_FACTOR, IMG_SIZE[1] // DOWNSAMPLE_FACTOR)

    for o in range(n):
        z = torch.randn((1, LATENT_DIM, latent_size[0], latent_size[1]), device=device)
        class_label = torch.tensor([o % 7], device=device)

        for i in range(T - 1, -1, -1):  # Reverse diffusion process
            t = torch.full((1,), i, device=device, dtype=torch.long)
            z = sample_timestep(z, class_label, t)
            z = torch.clamp(z, -1.0, 1.0)  # Keep in range [-1, 1]

        # Decode latent vector back into an image
        with torch.no_grad():
            image = decoder(z)

        # Convert tensor to PIL image and save
        image = show_tensor_image(image.detach().cpu())
        image.save(f"{joined}/class{o}_{tag}.png")

def load_model(model_path, num_classes, device="cuda"):
    """Load a trained Latent Diffusion Model."""
    model = LatentConditionalUnet(num_classes, latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LDM Model
model = load_model(MODEL_NAME, num_classes=7, device=device)

# Load Autoencoder (Decoder for Image Reconstruction)
autoencoder = Autoencoder(latent_dim=LATENT_DIM).to(device)
autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH, map_location=device))
autoencoder.eval()
decoder = autoencoder.decoder  # Use the decoder for final image reconstruction

# Generate images using the **Latent Diffusion Model**
generate_latent_images(tag=TAG)
