import torch
import math
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 200
PATH = 'data'
LEARNING_RATE = 1e-4
TIMESTEPS = 1000
IMG_WIDTH = 137
IMG_HEIGHT = 50

print(f"Using {DEVICE}")

def linear_beta_schedule(timesteps=TIMESTEPS, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


betas = linear_beta_schedule(timesteps=TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def pad_to_match(x1, x2):
    diff_height = x2.size(2) - x1.size(2)
    diff_width = x2.size(3) - x1.size(3)
    x1 = F.pad(x1, (0, diff_width, 0, diff_height))
    return x1

def get_loss(model, x_0, t):
    t = t[:x_0.shape[0]]
    x_noisy, noise = forward_diffusion_sample(x_0, t, DEVICE)
    noise_pred = model(x_noisy, t)
    noise_pred = pad_to_match(noise_pred, noise)
    
    l1_loss = F.l1_loss(noise, noise_pred)
    return l1_loss


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.attn(x)
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, upsample=False):
        super().__init__()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.upsample = upsample
        
        if upsample:
            self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1)
            self.transform = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,  padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, kernel_size=3,  stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,  padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.attention = AttentionBlock(out_channels)
    
    def forward(self, x, t):
        x = self.batch_norm1(self.activation(self.conv1(x)))
        
        time_emb = self.activation(self.time_mlp(t))
        time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
        x = x + time_emb
        
        x = self.batch_norm2(self.activation(self.conv2(x)))
        x = self.attention(x)
        
        return self.transform(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.embedding_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class UNet(nn.Module):
    def __init__(self, image_channels=1, base_channels=64, time_emb_dim=64):
        super().__init__()
        
        down_channels = [base_channels * 2**i for i in range(6)]
        up_channels = list(reversed(down_channels))
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.initial_conv = nn.Conv2d(image_channels, down_channels[0], kernel_size=3,  padding=1)
        
        self.down_blocks = nn.ModuleList([
            Block(down_channels[i], down_channels[i + 1], time_emb_dim)
            for i in range(len(down_channels) - 1)
        ])
        
        self.up_blocks = nn.ModuleList([
            Block(up_channels[i], up_channels[i + 1], time_emb_dim, upsample=True)
            for i in range(len(up_channels) - 1)
        ])
        
        self.final_conv = nn.Conv2d(up_channels[-1], image_channels, kernel_size=1)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.initial_conv(x)

        residuals = []
        for down in self.down_blocks:
            x = down(x, t_emb)
            residuals.append(x)

        for up in self.up_blocks:
            residual_x = residuals.pop()
            residual_x = pad_to_match(residual_x, x)  
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t_emb)

        return self.final_conv(x)

model = UNet()

def show_tensor_image(image):
    if len(image.shape) == 4:
        image = image[0]

    if image.shape[0] in [1, 3]:  # (C, H, W) format expected
        image = image.permute(1, 2, 0)  # Convert to (H, W, C)

    image = image.clamp(0, 1)  # Ensure pixel values are between 0-1
    plt.imshow(image.numpy(), cmap="gray" if image.shape[-1] == 1 else None)
    plt.axis("off")
    
@torch.no_grad()
def sample_timestep(x, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    noise_pred = model(x, t)
    
    # Ensure model output matches x's shape
    noise_pred = pad_to_match(noise_pred, x)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img = torch.randn((1, 1, IMG_HEIGHT, IMG_WIDTH), device=DEVICE)
    plt.figure(figsize=(5,5))
    plt.axis('off')
    num_images = 1
    stepsize = int(TIMESTEPS/num_images)

    for i in range(0,TIMESTEPS)[::-1]:
        t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
        img = sample_timestep(img, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.savefig('fig7.png')        
    #plt.show()
    
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

# some useful info about the dataset
print(f"Classes: {dataset.classes}")
print(f"Number of  samples: {len(dataset)}")

for i, (x, _) in enumerate(dataloader):
    print(f"Image shape: {x.shape}")
    break

def train(): 
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        # Wrap the dataloader with tqdm for progress display
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}", unit='batch') as pbar:
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
                
                loss = get_loss(model, batch[0], t)
                loss.backward()
                optimizer.step()

                # Update tqdm progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

    torch.save(model.state_dict(), "model7.pth")
    sample_plot_image()

train()