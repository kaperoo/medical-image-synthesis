import os
import sys
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from latentmodel import LatentConditionalUnet
import tqdm
from autoencoder import Autoencoder  # Import the pretrained autoencoder




PATH_TO_CHECKPOINT = "./latentcheckpoints"
PATH_TO_DATA = "../../../data/augmented_data"
# IMG_SIZE = (64, 144)
IMG_SIZE = (128, 288)
LEARING_RATE = 1e-3
EPOCHS = 300
BATCH_SIZE = 16
T = 4000

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
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    dataset = torchvision.datasets.ImageFolder(PATH_TO_DATA, transform=data_transform)
    return dataset

def get_loss(model, x_0, t, class_labels):
    """
    Compute the loss in latent space instead of pixel space.
    """
    z_0 = encoder(x_0)  # Encode image to latent space
    z_noisy, noise = forward_diffusion_sample(z_0, t, device)  # Add noise in latent space
    noise_pred = model(z_noisy, t, class_labels)  # Predict noise in latent space
    return F.l1_loss(noise, noise_pred)  # Compute L1 loss


def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e6  # Convert to MB
    reserved = torch.cuda.memory_reserved() / 1e6  # Convert to MB
    print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

if __name__ == "__main__":
    
    model = LatentConditionalUnet()
    data = load_transformed_dataset(adjust_image_size(IMG_SIZE))
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = EPOCHS# Load pretrained autoencoder
    autoencoder = Autoencoder(latent_dim=4).to(device)
    autoencoder.load_state_dict(torch.load("autoencoder.pth"))  # Load trained model
    autoencoder.eval()

    # Extract encoder and decoder
    encoder = autoencoder.encoder  # To encode images to latent space
    decoder = autoencoder.decoder  # To decode latent vectors back to images
    
    start_epoch = 0
    if len(sys.argv) > 1:
        if sys.argv[1] == "resume":
            checkpoint = torch.load(os.path.join(PATH_TO_CHECKPOINT, "state.pth"))
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']
            model.load_state_dict(torch.load(os.path.join(PATH_TO_CHECKPOINT, "model.pth")))
            print(f"Resuming training from epoch {start_epoch}")
        
        else:
            print("Invalid argument. Exiting...")
            sys.exit(1)
            
    for epoch in range(start_epoch, epochs):
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}", unit='batch') as pbar:
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                images, class_labels = batch[0].to(device), batch[1].to(device)

                # Convert images to latent space
                with torch.no_grad():  
                    z_0 = encoder(images)  # Encode image to latent space

                t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
                loss = get_loss(model, images, t, class_labels)

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

            # Save model checkpoint every 10 epochs
            if epoch % 10 == 0:
                print(f"Saving model to {PATH_TO_CHECKPOINT}")
                torch.save(model.state_dict(), os.path.join(PATH_TO_CHECKPOINT, "model.pth"))
                torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, os.path.join(PATH_TO_CHECKPOINT, "state.pth"))

    # Final save
    torch.save(model.state_dict(), "latentmodel.pth")
