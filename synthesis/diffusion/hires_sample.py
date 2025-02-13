import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
#from diffusion_model_cond_hires import SimpleUnet
from latenightmodel import SimpleUnet
#from model import SimpleUnet
import os

#MODEL_NAME = "model.pth"
#MODEL_NAME = "diffusion_model_cond_hires.pth"
MODEL_NAME = "latenightmodel.pth"
IMG_SIZE = (128,288)
T = 1000
TAG = "128x288-750e"

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def show_tensor_image(image):
    """
    Converts a PyTorch tensor to a PIL image, ensuring proper scaling.
    """
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min())),  # Scale from [-1, 1] to [0, 1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),  # Scale to [0, 255]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  # Convert to uint8
        transforms.ToPILImage(),
    ])

    # If batch dimension exists, take the first image
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    
    #plt.imshow(reverse_transforms(image), cmap='gray')
    return reverse_transforms(image)



# Define beta schedule
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
@torch.no_grad()
def sample_timestep(x, class_label, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, class_label) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 1, img_size[0], img_size[1]), device=device)
    plt.figure(figsize=(15,3))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)
    class_label = torch.tensor([0], device=device)
    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, class_label, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()

@torch.no_grad()
def plot_n_images(n=10):
    # generate 10 sample images and plot them
    img_size = IMG_SIZE
    img = torch.randn((10, 1, img_size[0], img_size[1]), device=device)
    plt.figure(figsize=(15,3))
    plt.axis('off')
    for o in range(n):
        class_label = torch.tensor([o % 7], device=device)
        for i in range(0,T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(img, class_label, t)
            # Edit: This is to maintain the natural range of the distribution
            img = torch.clamp(img, -1.0, 1.0)
            if i == 0:
                plt.subplot(1, n, o+1)
                show_tensor_image(img.detach().cpu())
    #plt.show()
    plt.savefig("no_eval_test.png")

@torch.no_grad()
def generate_images(n=7, tag='other'):
    save_dir = f"generated_images/{tag}"
    os.makedirs(save_dir, exist_ok=True)

    img_size = IMG_SIZE
    for o in range(n):
        img = torch.randn((1, 1, img_size[0], img_size[1]), device=device)
        class_label = torch.tensor([o % 7], device=device)

        for i in range(0, T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(img, class_label, t)
            img = torch.clamp(img, -1.0, 1.0)  # Keep in range [-1, 1]

        # Convert tensor to PIL and save
        image = show_tensor_image(img.detach().cpu())
        image.save(f"{save_dir}/class{o}_{tag}.png")


def load_model(model_path, num_classes, device="cuda"):
    """Load a saved diffusion model"""
    model = SimpleUnet(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    # model.eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(MODEL_NAME, num_classes=7, device=device)
# print(model)

#plot_n_images(7)
generate_images(tag=TAG)
