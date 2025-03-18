import os
import sys

sys.path.append(sys.argv[0])

import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from latentmodel import LatentConditionalUnet
import tqdm
from autoencoder import Autoencoder

PRINT = True
LOG = False
SAVE_PATH = "."
PATH_TO_CHECKPOINT = "./latentcheckpoints"
PATH_TO_DATA = "../../../data/augmented_data"
# IMG_SIZE = (64, 144)
# IMG_SIZE = (128, 288)
IMG_SIZE = (128, 352)
# IMG_SIZE = (208, 560)
LEARING_RATE = 1e-3
EPOCHS = 1000
BATCH_SIZE = 16
T = 1000

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


def forward_diffusion_sample(z_0, t, device="cpu"):
    """
    Takes a latent vector and a timestep as input and
    returns the noisy version of it.
    """
    noise = torch.randn_like(z_0)  # Generate Gaussian noise in latent space
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, z_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, z_0.shape
    )
    # Mean + variance
    return sqrt_alphas_cumprod_t.to(device) * z_0.to(
        device
    ) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
betas = linear_beta_schedule(timesteps=T)

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

def adjust_image_size(img_size):
    num_downsample = 4
    min_size = 2**num_downsample
    new_height = ((img_size[0] + min_size - 1) // min_size) * min_size
    new_width = ((img_size[1] + min_size - 1) // min_size) * min_size
    return (new_height, new_width)

def load_transformed_dataset(img_size=IMG_SIZE):
    data_transforms = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size[0], img_size[1])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Normalize([0.5], [0.5]),
        # transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    dataset = torchvision.datasets.ImageFolder(PATH_TO_DATA, transform=data_transform)
    return dataset

def get_loss(model, z_0, t, class_labels, device):
    """
    Compute the loss in latent space instead of pixel space.
    """
    z_noisy, noise = forward_diffusion_sample(z_0, t, device)  # Add noise in latent space
    noise_pred = model(z_noisy, t, class_labels)  # Predict noise in latent space
    return F.l1_loss(noise, noise_pred)  # Compute L1 loss
    # return F.mse_loss(noise, noise_pred)  # Compute MSE loss


def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e6  # Convert to MB
    reserved = torch.cuda.memory_reserved() / 1e6  # Convert to MB
    print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


@torch.no_grad()
def generate(idx='', device="cpu"):
    latent_dim = 4
    img_size = (IMG_SIZE[0] // 4, IMG_SIZE[1] // 4)
    class_labels = torch.arange(7, dtype=torch.long, device=device)
    img = torch.randn((7, latent_dim, img_size[0], img_size[1]), device=device)

    for i in range(0, T)[::-1]:
        t = torch.full((7,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, class_labels, t)
        img = torch.clamp(img, -1.0, 1.0)
        
    # loop over the images in the img and decode them
    new_img = torch.zeros((7, 1, IMG_SIZE[0], IMG_SIZE[1]), device=device)
    for i in range(7):
        z = img[i].unsqueeze(0)
        with torch.no_grad():
            image = decoder(z)
        new_img[i] = image
    
    fig_path = SAVE_PATH + "/generated/concat/"
    torchvision.utils.save_image(new_img, f"{fig_path}{idx}.png", normalize=True)

if __name__ == "__main__":
    
    model = LatentConditionalUnet()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device}")
    model.to(device)
    # optimizer = Adam(model.parameters(), lr=LEARING_RATE)
    optimizer = AdamW(model.parameters(), lr=LEARING_RATE)
    epochs = EPOCHS
    autoencoder = Autoencoder(latent_dim=4).to(device)    
    autoencoder.load_state_dict(torch.load(os.path.join(sys.path[0],"autoencoder.pth")))  # Load trained model
    autoencoder.eval()
    
    start_epoch = 0
    
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if i == 0:
                continue
            if arg == "--resume":
                save_path = os.path.join(SAVE_PATH, PATH_TO_CHECKPOINT)
                checkpoint = torch.load(os.path.join(save_path, "state.pth"))
                start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                loss = checkpoint['loss']
                model.load_state_dict(torch.load(os.path.join(save_path, "model.pth")))
                # print(f"Resuming training from epoch {start_epoch}")
            elif arg == "--save":
                SAVE_PATH = sys.argv[i+1]
            elif arg == "--data":
                PATH_TO_DATA = sys.argv[i+1]
            elif arg == "-p":
                PRINT = False
            elif arg == "-l":
                LOG = True
    
    data = load_transformed_dataset(adjust_image_size(IMG_SIZE))
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    encoder = autoencoder.encoder  # To encode images to latent space
    decoder = autoencoder.decoder  # To decode latent vectors back to images
            
    for epoch in range(start_epoch, epochs):
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}", unit='batch', disable= (not PRINT)) as pbar:
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                images, class_labels = batch[0].to(device), batch[1].to(device)

                # Convert images to latent space
                with torch.no_grad():  
                    z_0 = encoder(images)  # Encode image to latent space

                t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
                loss = get_loss(model, z_0, t, class_labels, device)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

            # Save model checkpoint every 10 epochs
            if epoch % 10 == 9 and epoch > 1:
                generate(idx=epoch, device=device)
                if PRINT:
                    print(f"Saving model to {PATH_TO_CHECKPOINT}")
                save_path = os.path.join(SAVE_PATH, PATH_TO_CHECKPOINT)
                torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
                torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, os.path.join(save_path, "state.pth"))
            
            if LOG:
                # see if logfile exists
                log_path = os.path.join(SAVE_PATH, "log.txt")
                if not os.path.exists(log_path):
                    with open(log_path, "w") as f:
                        f.write(f"Epoch {epoch}, Loss: {loss.item()}\n")
                else:
                    with open(log_path, "a") as f:
                        f.write(f"Epoch {epoch}, Loss: {loss.item()}\n")
                

    # Final save
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "latentmodel.pth"))
