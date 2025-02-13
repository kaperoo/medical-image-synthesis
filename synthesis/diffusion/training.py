import os
import sys
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import ConditionalUnet

PATH_TO_CHECKPOINT = "./checkpoints"
PATH_TO_DATA = "../../data/augmented_data"
IMG_SIZE = (64, 144)
LEARING_RATE = 1e-3
EPOCHS = 300
BATCH_SIZE = 8
T = 2000

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
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(
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
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t, class_labels)
    return F.l1_loss(noise, noise_pred)

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e6  # Convert to MB
    reserved = torch.cuda.memory_reserved() / 1e6  # Convert to MB
    print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

# TODO: ADD CHECKPOINTING
if __name__ == "__main__":
    
    model = ConditionalUnet()
    data = load_transformed_dataset(adjust_image_size(IMG_SIZE))
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = EPOCHS
    
    start_epoch = 0
    # if sys.argv > 1:


    for epoch in range(start_epoch, epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            images, class_labels = batch[0].to(device), batch[1].to(device)

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, images, t, class_labels)
            loss.backward()
            optimizer.step()
            #torch.cuda.empty_cache()

            # if epoch % 5 == 0 and step == 0:
            if step % 50 == 0:
                # print_gpu_memory()
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                # sample_plot_image()

        if epoch % 10 == 0:
            print(f"Saving model to {PATH_TO_CHECKPOINT}")
            torch.save(model.state_dict(), os.path.join(PATH_TO_CHECKPOINT, "model.pth"))
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, os.path.join(PATH_TO_CHECKPOINT, "state.pth"))

    torch.save(model.state_dict(), "model.pth")
