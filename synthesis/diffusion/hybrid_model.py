import torch
import math
import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms

# --- SETUP ---
# general
ITERATION = '60'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH = 'data'
RESULTS_PATH = f"results{ITERATION}"
os.makedirs(RESULTS_PATH, exist_ok=True)
FIG_PATH = f"{RESULTS_PATH}/fig{ITERATION}_"
MODEL_PATH = f"{RESULTS_PATH}/model{ITERATION}_"

# data info
NUM_CLASSES = 7
IMG_WIDTH = 128
IMG_HEIGHT = 64

# hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LEARNING_RATE_DISCR = 1e-5
TIMESTEPS = 300
TIME_EMBEDDING_DIM = 128
BASE_CHANNELS = 128
NUM_LAYERS = 5
CLASS_EMB_WEIGHT = 5
ADV_LOSS_WEIGHT = 0.2

# epochs
EPOCHS = 100

print(f"Using {DEVICE}")

# --- NOISE SCHEDULER ---
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

# --- HELPER FUNCTIONS ---
def get_loss(model, x_0, t, class_labels):
    t = t[:x_0.shape[0]]
    x_noisy, noise = forward_diffusion_sample(x_0, t, DEVICE)
    noise_pred = model(x_noisy, t, class_labels)
    return F.l1_loss(noise, noise_pred)

def get_adversarial_loss(discr, fake_images, loss_d, device=DEVICE):
    fake_preds = discr(fake_images)
    real_labels = torch.ones_like(fake_preds).to(device)
    return loss_d(fake_preds, real_labels)

def load_data(): 
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_HEIGHT-16, IMG_WIDTH)),
        transforms.Pad((0, 8)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    dataset = datasets.ImageFolder(PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

    print(f"Classes: {dataset.classes}")
    print(f"Number of  samples: {len(dataset)}")

    for i, (x, _) in enumerate(dataloader):
        print(f"Image shape: {x.shape}")
        break
    return dataloader

def load_model(model, path):
    model.to(DEVICE)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=DEVICE))
    return model

# --- DISCRIMINATOR ---
class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)


# --- UNET ---
class AttentionBlock(nn.Module):
    # Convolutional Block Attention Module (CBAM)
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        
        # Channel Attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        attn_channel = self.channel_attn(x) * x
        
        # Spatial Attention
        avg_out = torch.mean(attn_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(attn_channel, dim=1, keepdim=True)
        attn_spatial = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        
        return attn_channel * attn_spatial 
    
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
            self.transform = nn.Conv2d(out_channels, out_channels, kernel_size=4,  stride=2, padding=1)
        
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
    def __init__(self, 
                 image_channels=1, 
                 base_channels=BASE_CHANNELS, 
                 time_emb_dim=TIME_EMBEDDING_DIM, 
                 num_classes=NUM_CLASSES,
                 num_layers=NUM_LAYERS):
        super().__init__()
        
        down_channels = [base_channels * 2**i for i in range(num_layers)]
        up_channels = list(reversed(down_channels))
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.class_embedding = nn.Embedding(num_classes, time_emb_dim)
        
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
    
    def forward(self, x, t, class_label):
        t_emb = self.time_mlp(t)
        class_emb = self.class_embedding(class_label)
        t_emb = t_emb + class_emb * CLASS_EMB_WEIGHT

        x = self.initial_conv(x)

        residuals = []
        for down in self.down_blocks:
            x = down(x, t_emb)
            residuals.append(x)

        for up in self.up_blocks:
            residual_x = residuals.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t_emb)

        return self.final_conv(x)

# --- SAMPLING --- 
def reverse_image (image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min())),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    
    return reverse_transforms(image)

@torch.no_grad()
def sample_timestep(x, class_label, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_output = model(x, t, class_label)
    
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
def generate_images(idx=''):
    print('\nGenerating images...')
    names = ['AMD', 'DME', 'ERM', 'NO', 'RAO', 'RVO', 'VID']
    img_size =  (IMG_HEIGHT, IMG_WIDTH)
    
    for class_idx in range(NUM_CLASSES):
        img = torch.randn((1, 1, img_size[0], img_size[1]), device=DEVICE)
        class_label = torch.tensor([class_idx % 7], device=DEVICE)

        for i in range(0, TIMESTEPS)[::-1]:
            t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
            img = sample_timestep(img, class_label, t)
            img = torch.clamp(img, -1.0, 1.0)

        image = reverse_image(img.detach().cpu())
        image.save(f"{FIG_PATH}{names[class_idx]}_{idx}.png")
    print('Done.')
    
# generate for discriminator input
@torch.no_grad()
def generate_fake_images(batch_size, class_labels):
    img_size = (IMG_HEIGHT, IMG_WIDTH)
    img = torch.randn((batch_size, 1, img_size[0], img_size[1]), device=DEVICE)

    for i in range(TIMESTEPS - 1, -1, -1): 
        t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE,).long()
        img = sample_timestep(img, class_labels, t)
        img = torch.clamp(img, -1.0, 1.0)
    
    return img 

# --- TRAINING ---
def train(discr_steps=5):
    model.to(DEVICE)
    discr.to(DEVICE)
    
    model.train()
    discr.train()
    
    for epoch in range(EPOCHS):
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}", unit='batch') as pbar:
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                optimizer_d.zero_grad()
                
                images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                
                if step % 10 == 0 or fake_images_cache is None:
                    fake_images_cache = generate_fake_images(images.size(0), class_labels)
                fake_images = fake_images_cache.detach()
                
                # Diffusion loss
                t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
                diffusion_loss = get_loss(model, images, t, class_labels)
                diffusion_loss.backward()
                optimizer.step()
                
                # Discriminator loss
                for _ in range(discr_steps):
                    real_labels = torch.ones(images.size(0), 1).to(DEVICE)
                    fake_labels = torch.zeros(fake_images.size(0), 1).to(DEVICE)

                    real_preds = discr(images)
                    real_loss = loss_d(real_preds, real_labels)

                    fake_preds = discr(fake_images.detach())
                    fake_loss = loss_d(fake_preds, fake_labels)

                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                optimizer_d.step()
                    
                adv_loss = get_adversarial_loss(discr, fake_images.detach(), loss_d, DEVICE)
                total_loss = diffusion_loss + adv_loss * ADV_LOSS_WEIGHT 
                
                pbar.set_postfix(diffusion_loss=diffusion_loss.item(),
                                 adv_loss=adv_loss.item(),
                                 total_loss=total_loss.item())
                pbar.update(1)

    torch.save(model.state_dict(), f"{MODEL_PATH}discr.pth")
    torch.save(discr.state_dict(), f"{MODEL_PATH}discr.pth")

if __name__ == "__main__":
    dataloader = load_data()
    
    model = UNet()
    discr = Discriminator()
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_d = Adam(model.parameters(), lr=LEARNING_RATE_DISCR)
    loss_d = nn.BCELoss()
    
    print('\nTraining...')
    train()
    
    #model = load_model(model, f"{MODEL_PATH}diff.pth")
    #model = load_model(model, f"{MODEL_PATH}discr.pth")
    
    generate_images('1')
    generate_images('2')