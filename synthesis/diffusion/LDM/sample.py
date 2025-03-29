import sys

sys.path.append(sys.argv[0])

import torch
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from latentmodel import LatentConditionalUnet
import tqdm
from vae import Autoencoder, Encoder, Decoder
from latenttraining import linear_beta_schedule, get_index_from_list

TAG = "ANNEALTEST"

# Define beta schedule
betas = linear_beta_schedule(timesteps=1000)

# Pre-calculate different terms for closed form
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

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

    if torch.all(t == 0):
        return model_mean
    else:
        noise = torch.randn_like(z)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    
@torch.no_grad()
def generate(idx='', device="cpu", autoencoder=None):
    latent_dim = 4
    img_size = (128 // 4, 352 // 4)
    class_labels = torch.arange(7, dtype=torch.long, device=device)
    img = torch.randn((7, latent_dim, img_size[0], img_size[1]), device=device)

    for i in range(0, 1000)[::-1]:
        t = torch.full((7,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, class_labels, t)
        img = torch.clamp(img, -1.0, 1.0)
        
    # loop over the images in the img and decode them
    new_img = torch.zeros((7, 1, 128, 352), device=device)
    for i in range(7):
        z = img[i].unsqueeze(0)
        with torch.no_grad():
            image = autoencoder.module.decode(z / 0.18215)
        new_img[i] = image
    
    print(new_img.shape)
    torchvision.utils.save_image(new_img, f"{idx}b.png", normalize=True)

def load_model(model_path, num_classes, device="cuda"):
    """Load a trained Latent Diffusion Model."""
    model = LatentConditionalUnet()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LDM Model
model = load_model("./latentmodel.pth", num_classes=7, device=device)

# Load Autoencoder (Decoder for Image Reconstruction)
encoder = Encoder(
    channels=128,
    channel_multipliers=[1, 2, 4],
    n_resnet_blocks=2,
    in_channels=1,
    z_channels=4
)
    
decoder = Decoder(
    channels=128,
    channel_multipliers=[1, 2, 4],
    n_resnet_blocks=2,
    out_channels=1,
    z_channels=4
)

autoencoder = Autoencoder(
    encoder=encoder,
    decoder=decoder,
    emb_channels=4,
    z_channels=4
)
autoencoder = torch.nn.DataParallel(autoencoder)
autoencoder.to(device)
autoencoder.load_state_dict(torch.load("./vae.pth"))

# model = torch.nn.DataParallel(model)

autoencoder.eval()
model.eval()
# Generate images using the **Latent Diffusion Model**
for i in range(3):
    generate(idx=i, device=device, autoencoder=autoencoder)