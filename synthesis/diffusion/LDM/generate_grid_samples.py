import os
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import torchvision.transforms as transforms
from latentmodel import LatentConditionalUnet
from autoencoder import Autoencoder

# Configuration
LATENT_MODEL_PATH = "./latentmodel.pth"  # Update with your model path
AUTOENCODER_PATH = "./autoencoder.pth"    # Update with your autoencoder path
TAG = "LDM_test4"
OUTPUT_DIR = "./generated/"+TAG
NUM_CLASSES = 7
LATENT_DIM = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 32, 88  # (128//4, 352//4) based on your encoder architecture

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_models():
    # Load the latent diffusion model
    model = LatentConditionalUnet(num_classes=NUM_CLASSES, latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(LATENT_MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Load the autoencoder
    autoencoder = Autoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH, map_location=DEVICE))
    autoencoder.eval()
    
    return model, autoencoder.decoder

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def sample_from_model(model, class_label, device=DEVICE, steps=1000):
    # Parameters for sampling
    betas = linear_beta_schedule(timesteps=steps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    # Start with random noise
    x_shape = (1, LATENT_DIM, H, W)
    sample = torch.randn(x_shape, device=device)
    
    # Setup class tensor
    class_tensor = torch.tensor([class_label], device=device)
    
    # Iterative sampling - reverse diffusion process
    for i in reversed(range(0, steps)):
        # Get timestep
        t = torch.full((1,), i, device=device, dtype=torch.long)
        
        # Get alpha values for this timestep
        alpha = alphas[i]
        alpha_cumprod = alphas_cumprod[i]
        alpha_cumprod_prev = alphas_cumprod_prev[i] if i > 0 else torch.tensor(1.0)
        
        # Predict noise
        with torch.no_grad():
            predicted_noise = model(sample, t, class_tensor)
            
        # Mean component
        mean = (sample - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
        
        # If last step, don't add more noise
        if i > 0:
            noise = torch.randn_like(sample)
            variance = torch.sqrt(posterior_variance[i])
            sample = mean + variance * noise
        else:
            sample = mean
            
    return sample

def generate_class_grid():
    # Load models
    model, decoder = load_models()
    
    # List to store all generated images
    all_images = []
    
    # Generate one image per class
    for class_idx in range(NUM_CLASSES):
        print(f"Generating image for class {class_idx}...")
        
        # Generate latent from model
        latent = sample_from_model(model, class_idx)
        
        # Decode to image
        with torch.no_grad():
            generated_image = decoder(latent)
        
        # Normalize to [0,1] range
        generated_image = (generated_image + 1) / 2.0
        generated_image = torch.clamp(generated_image, 0, 1)
        
        # Add to our collection
        all_images.append(generated_image)
    
    # Stack all images into a single tensor
    all_images_tensor = torch.cat(all_images, dim=0)
    
    # Create a grid of images
    grid = vutils.make_grid(all_images_tensor, nrow=4, padding=10, pad_value=1)
    
    # Save the grid
    grid_path = os.path.join(OUTPUT_DIR, "all_classes_grid.png")
    vutils.save_image(grid, grid_path)
    
    print(f"Saved grid with all classes to {grid_path}")
    
    # Save individual images too (optional)
    for i, img in enumerate(all_images):
        img_path = os.path.join(OUTPUT_DIR, f"class_{i}.png")
        vutils.save_image(img, img_path)

if __name__ == "__main__":
    generate_class_grid()