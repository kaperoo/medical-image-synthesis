print('Initialising...')
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
from sklearn.utils.class_weight import compute_class_weight

# --- SETUP ---
# general
ITERATION = '69'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH = 'data'
RESULTS_PATH = f"results{ITERATION}"
os.makedirs(RESULTS_PATH, exist_ok=True)
FIG_PATH = f"{RESULTS_PATH}/fig{ITERATION}_"
MODEL_PATH = f"{RESULTS_PATH}/model{ITERATION}.pth"

# data info
NUM_CLASSES = 7
IMG_WIDTH = 128
IMG_HEIGHT = 64

# hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TIMESTEPS = 1000
TIME_EMBEDDING_DIM = 128
BASE_CHANNELS = 128
CLASS_EMB_WEIGHT = 0.7
NUM_LAYERS = 5
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
    
# --- HELPER FUNCTIONS ---
def get_loss(model, x_0, t, class_labels, class_weights):
    loss = nn.SmoothL1Loss(beta=0.1) 
    
    t = t[:x_0.shape[0]]
    x_noisy, noise = forward_diffusion_sample(x_0, t, DEVICE)
    noise_pred = model(x_noisy, t, class_labels)
    
    per_sample_loss = loss(noise, noise_pred)
    weights = class_weights[class_labels].view(-1,1,1,1)
    weighted_loss = (per_sample_loss * weights).mean()

    return weighted_loss

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
    
# --- CBAM ATTENTION ---
# src: https://arxiv.org/abs/1807.06521
#      https://github.com/Jongchan/attention-module/

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
    
# --- MODEL ARCHITECTURE ---
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
        self.attention = CBAM(out_channels)
    
    def forward(self, x, t):
        x = self.batch_norm1(self.activation(self.conv1(x)))
        
        time_emb = self.activation(self.time_mlp(t))
        time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
        x = x + time_emb
        
        x = self.batch_norm2(self.activation(self.conv2(x)))
        
        if self.upsample:
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
        
        self.middle_attention = CBAM(down_channels[-1])
        
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
        
        x = self.middle_attention(x)
        
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
def generate_and_save(idx=''):
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
        image.save(FIG_PATH + names[class_idx] + idx + '.png')


# --- TRAINING ---
def train_diffusion(): 
    model.train()
    labels = np.concatenate([batch[1].numpy() for batch in dataloader])
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f'Class weights: {class_weights}')
    
    for epoch in range(EPOCHS):
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}", unit='batch') as pbar:
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                
                t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
                loss = get_loss(model, images, t, class_labels, class_weights)
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                
    torch.save(model.state_dict(), f"{MODEL_PATH}.pth")

if __name__ == "__main__":
    dataloader = load_data()
    
    model = UNet()
    model.to(DEVICE)    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    print('\nTraining diffusion model...')
    train_diffusion()
    #model = load_model(model, f"{MODEL_PATH}.pth")
    
    
    print('\nGenerating images...')
    generate_and_save('1')
    generate_and_save('2')
    print('Done.')
