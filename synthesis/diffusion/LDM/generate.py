import os
import argparse
import torch
import torchvision
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# Import your models
from latentmodel import LatentConditionalUnet
from vae import Autoencoder, Encoder, Decoder

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Latent Diffusion Model")
    parser.add_argument("--save_dir", type=str, default="./generated", help="Directory to save generated images")
    parser.add_argument("--model_path", type=str, default="./latentmodel.pth", help="Path to pretrained LDM")
    parser.add_argument("--vae_path", type=str, default="./vae.pth", help="Path to pretrained VAE")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--num_images", type=int, default=21, help="Number of images to generate")
    parser.add_argument("--class_id", type=int, default=None, help="Class ID to generate (None for all classes)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--lsf", type=float, default=0.18215, help="Latent scaling factor (try 0.15 for better shape quality)")
    parser.add_argument("--img_size", type=str, default="128,352", help="Image size as height,width")
    return parser.parse_args()

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def sample_timestep(model, z, class_label, t):
    """
    Performs a reverse diffusion step without guidance
    """
    betas_t = get_index_from_list(betas, t, z.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, z.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, z.shape)
    
    # Get noise prediction from model
    if isinstance(model, torch.nn.DataParallel):
        noise_pred = model.module(z, t, class_label)
    else:
        noise_pred = model(z, t, class_label)
    
    # Compute mean for reverse process
    model_mean = sqrt_recip_alphas_t * (
        z - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
    )
    
    posterior_variance_t = get_index_from_list(posterior_variance, t, z.shape)
    
    if torch.all(t == 0):
        return model_mean
    else:
        noise = torch.randn_like(z)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def main():
    args = parse_args()
    
    # Parse image size
    h, w = map(int, args.img_size.split(','))
    img_size = (h, w)
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize models
    print("Loading models...")
    
    # Load LDM
    model = LatentConditionalUnet()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load VAE
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
    autoencoder.load_state_dict(torch.load(args.vae_path, map_location=device))
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    # Define latent size (assuming downsampling by factor of 4 as in your code)
    latent_size = (img_size[0] // 4, img_size[1] // 4)
    
    # Pre-calculate beta schedule
    global betas, alphas, alphas_cumprod, sqrt_recip_alphas
    global sqrt_one_minus_alphas_cumprod, posterior_variance, alphas_cumprod_prev
    
    T = args.steps
    betas = linear_beta_schedule(timesteps=T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    # Define classes to generate
    if args.class_id is not None:
        class_ids = [args.class_id]
    else:
        class_ids = list(range(7))  # NUM_CLASSES=7 in your model
    
    # Generate images
    print(f"Generating {args.num_images} images...")
    
    image_count = 0
    for class_id in class_ids:
        num_images_per_class = args.num_images // len(class_ids) if args.class_id is None else args.num_images
        
        remaining = num_images_per_class
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            remaining -= batch_size
            
            # Generate random latent vectors
            latents = torch.randn(batch_size, 4, latent_size[0], latent_size[1], device=device)
            
            # Create class labels
            class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
            
            # Sample through diffusion process
            print(f"Sampling for class {class_id}, batch of {batch_size} images...")
            
            # Progressively denoise the latents
            for i in tqdm(reversed(range(T)), desc="Sampling timesteps"):
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                latents = sample_timestep(
                    model=model, 
                    z=latents, 
                    class_label=class_labels, 
                    t=timesteps
                )
                latents = torch.clamp(latents, -1.0, 1.0)
                
            # Decode latents to images
            with torch.no_grad():
                if isinstance(autoencoder, torch.nn.DataParallel):
                    images = autoencoder.module.decode(latents / args.lsf)
                else:
                    images = autoencoder.decode(latents / args.lsf)
            
            # Save individual images
            for j in range(batch_size):
                image_path = os.path.join(args.save_dir, f"{class_id}/class_{class_id}_sample_{image_count}.png")
                torchvision.utils.save_image(images[j], image_path, normalize=True)
                image_count += 1
            
            # Save grid of all images in this batch
            # grid_path = os.path.join(args.save_dir, f"{class_id}/class_{class_id}_grid_{image_count-batch_size}-{image_count-1}.png")
            # torchvision.utils.save_image(images, grid_path, nrow=4, normalize=True)
    
    print(f"Generated {image_count} images in {args.save_dir}")

if __name__ == "__main__":
    main()
