import os
import sys
import argparse

sys.path.append(sys.argv[0])

import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from latentmodel import LatentConditionalUnet
import tqdm
from vae import Autoencoder, Encoder, Decoder

def parse_args():
    parser = argparse.ArgumentParser(description="Train a latent model")
    parser.add_argument("--data", type=str, default="../../../data/augmented_data", help="Path to dataset")
    parser.add_argument("--save", type=str, default=".", help="Path to save model")
    parser.add_argument("--tag", type=str, default="latentcheckpoints", help="Tag for saving model")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("-p", action="store_true", help="Disable print")
    parser.add_argument("-l", action="store_true", help="Enable log")
    parser.add_argument("--lsf", type=float, default=0.18215, help="Latent scaling factor")
    return parser.parse_args()

IMG_SIZE = (128, 352)
# IMG_SIZE = (208, 560)
LEARING_RATE = 1e-3
EPOCHS = 1000
BATCH_SIZE = 8
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

    dataset = torchvision.datasets.ImageFolder(args.data, transform=data_transform)
    return dataset

def get_loss(model, z_0, t, class_labels, device):
    """
    Compute the loss in latent space instead of pixel space.
    """
    z_noisy, noise = forward_diffusion_sample(z_0, t, device)  # Add noise in latent space
    noise_pred = model(z_noisy, t, class_labels)  # Predict noise in latent space
    #TODO: test smooth_l1_loss
    # return F.smooth_l1_loss(noise, noise_pred, beta=0.1)  # Compute L1 loss
    return F.l1_loss(noise, noise_pred)  # Compute L1 loss
    # return F.mse_loss(noise, noise_pred)  # Compute MSE loss


def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e6  # Convert to MB
    reserved = torch.cuda.memory_reserved() / 1e6  # Convert to MB
    print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


@torch.no_grad()
def generate(idx='', device="cpu", autoencoder=None):
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
            image = autoencoder.module.decode(z / args.lsf)
        new_img[i] = image
    
    fig_path = args.save + f"/generated/{args.tag}/"
    os.makedirs(fig_path, exist_ok=True)
    torchvision.utils.save_image(new_img, f"{fig_path}{idx}.png", normalize=True)


if __name__ == "__main__":
    
    args = parse_args()
    
    model = LatentConditionalUnet()
    
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
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        autoencoder = torch.nn.DataParallel(autoencoder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device}")
    
    model.to(device)
    autoencoder.to(device)
    
    autoencoder.load_state_dict(torch.load(os.path.join(args.save,"vae.pth")))  # Load trained model
    
    optimizer = AdamW(model.parameters(), lr=LEARING_RATE)
    scheduler = CosineAnnealingLR(optimizer, 500, eta_min=1e-6)
    epochs = EPOCHS

    autoencoder.eval()
    
    start_epoch = 0
    
    os.makedirs(os.path.join(args.save, args.tag), exist_ok=True)
    if args.resume:
        save_path = os.path.join(args.save, args.tag)
        checkpoint = torch.load(os.path.join(save_path, "state.pth"))
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        model.load_state_dict(torch.load(os.path.join(save_path, "model.pth")))
    
    data = load_transformed_dataset(adjust_image_size(IMG_SIZE))
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            
    for epoch in range(start_epoch, epochs):
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}", unit='batch', disable= (args.p)) as pbar:
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                images, class_labels = batch[0].to(device), batch[1].to(device)

                # Convert images to latent space
                with torch.no_grad():  
                    z_0 = args.lsf * autoencoder.module.encode(images).sample()

                t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
                loss = get_loss(model, z_0, t, class_labels, device)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

            # Save model checkpoint every 10 epochs
            if epoch % 10 == 9 and epoch > 1:
                generate(idx=epoch+1, device=device, autoencoder=autoencoder)
                save_path = os.path.join(args.save, args.tag)
                if (epoch +1 ) % 100 == 0:
                    torch.save(model.state_dict(), os.path.join(save_path, f"model_{epoch+1}.pth")) 
                else:
                    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
                torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, os.path.join(save_path, "state.pth"))
            
            if args.l:
                # see if logfile exists
                log_path = os.path.join(args.save, f"{args.tag}/log.txt")
                if not os.path.exists(log_path):
                    with open(log_path, "w") as f:
                        f.write(f"Epoch {epoch}, Loss: {loss.item()}\n")
                else:
                    with open(log_path, "a") as f:
                        f.write(f"Epoch {epoch}, Loss: {loss.item()}\n")
        scheduler.step()
                

    # Final save
    torch.save(model.state_dict(), os.path.join(args.save, f"{args.tag}/latentmodel.pth"))
