import torchvision

from models.discriminator import *
from models.unet import *
from util.config import *
from util.noise_scheduler import *
from util.helper_functions import *

# --- SAMPLING ---
@torch.no_grad()
def sample_timestep(model, x, class_label, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_output = model(x, t, class_label, CLASS_EMB_WEIGHT)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if torch.all(t == 0):
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def generate_and_save(model, idx=''):
    class_labels = torch.arange(NUM_CLASSES, dtype=torch.long, device=DEVICE)
    img_size = (IMG_HEIGHT, IMG_WIDTH)
    img = torch.randn((NUM_CLASSES, 1, img_size[0], img_size[1]), device=DEVICE)

    for i in range(0, TIMESTEPS)[::-1]:
        t = torch.full((NUM_CLASSES,), i, device=DEVICE, dtype=torch.long)
        img = sample_timestep(model, img, class_labels, t)
        img = torch.clamp(img, -1.0, 1.0)
    
    filename = f"{FIG_PATH}{idx}.png"
    torchvision.utils.save_image(img, filename, normalize=True)
    log(f'Saved {filename}')

@torch.no_grad()
def generate_fake_images(model, batch_size, class_labels):
    img_size = (IMG_HEIGHT, IMG_WIDTH)
    img = torch.randn((batch_size, 1, img_size[0], img_size[1]), device=DEVICE)

    for i in range(0, TIMESTEPS)[::-1]:
        t = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
        img = sample_timestep(model, img, class_labels, t)
        img = torch.clamp(img, -1.0, 1.0)
    return img 