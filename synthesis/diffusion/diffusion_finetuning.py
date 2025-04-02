import torch
import math
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight

# --- SETUP ---
# general
ITERATION = '106'
OUTPUT_TO_FILE = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH = 'data'
RESULTS_PATH = f"results{ITERATION}"
OUTPUTS_PATH = f"results{ITERATION}/outputs"
MODELS_PATH = f"results{ITERATION}/models"

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(OUTPUTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

FIG_PATH = f"{OUTPUTS_PATH}/fig{ITERATION}_"
MODEL_PATH = f"{MODELS_PATH}/model{ITERATION}"
LOGS_PATH = f"{OUTPUTS_PATH}/logs{ITERATION}.txt"

# data info
NUM_CLASSES = 7
IMG_WIDTH = 256
IMG_HEIGHT_SCALED = 94
IMG_HEIGHT = 128
PADDING = 17

# hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LEARNING_RATE_DISCR = 1e-4
TIMESTEPS = 1000
TIME_EMBEDDING_DIM = 128
BASE_CHANNELS = 128
CLASS_EMB_WEIGHT = 2
NUM_LAYERS = 5

CLASS_FREE_EPOCHS = 100
CLASS_EMB_EPOCHS = 300
DISCR_EPOCHS = 10
FINETUNE_EPOCHS = 100

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
    
# --- HELPER FUNCTIONS ---
def log(string):
    if not OUTPUT_TO_FILE:
        print(string)
    else:
        file = open(LOGS_PATH, "a")
        file.write(string)
        file.write('\n')
        file.close()

def get_loss(model, x_0, t, class_labels, class_weights):
    loss = nn.SmoothL1Loss(beta=0.1) 
    
    t = t[:x_0.shape[0]]
    x_noisy, noise = forward_diffusion_sample(x_0, t, DEVICE)
    noise_pred = model(x_noisy, t, class_labels, CLASS_EMB_WEIGHT)
    
    per_sample_loss = loss(noise, noise_pred)
    weights = class_weights[class_labels].view(-1,1,1,1)
    weighted_loss = (per_sample_loss * weights).mean()

    return weighted_loss

def get_adversarial_loss(discr, fake_images, loss_d, device=DEVICE):
    fake_preds = discr(fake_images)
    real_labels = torch.full_like(fake_preds, 0.9, device=device)
    return loss_d(fake_preds, real_labels)

def load_data(): 
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_HEIGHT_SCALED, IMG_WIDTH)),
        transforms.Pad((0, PADDING)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    dataset = datasets.ImageFolder(PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    return dataloader

def get_class_weights (dataloader):
    labels = np.concatenate([batch[1].numpy() for batch in dataloader])
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    return class_weights
    
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
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)
    
# Multihead Self Attention
class MSA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1) 

        x = self.norm(x)
        x, _ = self.attn(x, x, x)

        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

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
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.transform = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3,  padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2)
            )
        
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
        
        self.middle_attention = MSA(down_channels[-1])
        
        self.up_blocks = nn.ModuleList([
            Block(up_channels[i], up_channels[i + 1], time_emb_dim, upsample=True)
            for i in range(len(up_channels) - 1)
        ])
        
        self.final_conv = nn.Conv2d(up_channels[-1], image_channels, kernel_size=1)
    
    def forward(self, x, t, class_label, weight):
        t_emb = self.time_mlp(t)
        class_emb = self.class_embedding(class_label)
        t_emb = t_emb + class_emb * weight
        
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
    class_labels = torch.arange(NUM_CLASSES, dtype=torch.long, device=DEVICE)
    img_size = (IMG_HEIGHT, IMG_WIDTH)
    img = torch.randn((NUM_CLASSES, 1, img_size[0], img_size[1]), device=DEVICE)

    for i in range(0, TIMESTEPS)[::-1]:
        t = torch.full((NUM_CLASSES,), i, device=DEVICE, dtype=torch.long)
        img = sample_timestep(img, class_labels, t)
        img = torch.clamp(img, -1.0, 1.0)
    
    filename = f"{FIG_PATH}{idx}.png"
    torchvision.utils.save_image(img, filename, normalize=True)
    log(f'Saved {filename}')

@torch.no_grad()
def generate_fake_images(batch_size, class_labels):
    img_size = (IMG_HEIGHT, IMG_WIDTH)
    img = torch.randn((batch_size, 1, img_size[0], img_size[1]), device=DEVICE)

    for i in range(0, TIMESTEPS)[::-1]:
        t = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
        img = sample_timestep(img, class_labels, t)
        img = torch.clamp(img, -1.0, 1.0)
    return img 

# --- TRAINING ---
def train_diffusion_class_free(): 
    CLASS_EMB_WEIGHT = 0.0
    filename = f"{MODEL_PATH}_cf.pth"
    model.train()
    
    for epoch in range(CLASS_FREE_EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
                
            images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                
            t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
            loss = get_loss(model, images, t, class_labels, class_weights)
            loss.backward()
            optimizer.step()
                
        if (OUTPUT_TO_FILE):
            log(f"Epoch {epoch + 1}/{CLASS_FREE_EPOCHS}, Loss: {loss.item()}")      

        if ((epoch + 1) % 10 == 0):
           generate_and_save(f'step1_epoch{epoch + 1}')
           torch.save(model.state_dict(), filename)
           log(f"Saved checkpoint.")
                
    torch.save(model.state_dict(), filename)
    log(f"Saved {filename}")

# --- TRAINING ---
def train_diffusion_class(): 
    CLASS_EMB_WEIGHT = 2.0
    filename = f"{MODEL_PATH}_ce.pth"
    model.train()
    
    for epoch in range(CLASS_EMB_EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
                
            images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                
            t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
            loss = get_loss(model, images, t, class_labels, class_weights)
            loss.backward()
            optimizer.step()
                
        if (OUTPUT_TO_FILE):
            log(f"Epoch {epoch + 1}/{CLASS_EMB_EPOCHS}, Loss: {loss.item()}")      

        if ((epoch + 1) % 10 == 0):
           generate_and_save(f'step2_epoch{epoch + 1}')
           torch.save(model.state_dict(), filename)
           log(f"Saved checkpoint.")
                
    torch.save(model.state_dict(), filename)
    log(f"Saved {filename}")
    
def train_discriminator():
    filename = f'{MODEL_PATH}_discr.pth'
    discr.train()
    
    for epoch in range(DISCR_EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer_d.zero_grad()
            images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # Generate Fake Images
            if step % 50 == 0 or fake_images_cache is None:
                fake_images_cache = generate_fake_images(images.size(0), class_labels)
            fake_images = fake_images_cache.detach()

            # Compute discriminator loss
            optimizer_d.zero_grad()
                
            real_labels = torch.full((images.size(0), 1),  1.).to(DEVICE)
            fake_labels = torch.full((images.size(0), 1),  0.).to(DEVICE)

            real_preds = discr(images)
            real_loss = loss_d(real_preds[:images.size(0)], real_labels[:images.size(0)])

            fake_preds = discr(fake_images.detach())
            fake_loss = loss_d(fake_preds[:images.size(0)], fake_labels[:images.size(0)])

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()
                
                
        if (OUTPUT_TO_FILE):
            log(f"Epoch {epoch + 1}/{DISCR_EPOCHS}, Loss: {d_loss.item()}")    
            
    torch.save(discr.state_dict(), filename)
    log(f'Saved {filename}')
    
def finetune():
    filename_diff = f"{MODEL_PATH}.pth"
    model.train()
    discr.eval()
    
    for epoch in range(FINETUNE_EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            #optimizer_d.zero_grad()
            
            images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (images.size(0),), device=DEVICE).long()
            
            if step % 50 == 0 or fake_images_cache is None:
                fake_images_cache = generate_fake_images(images.size(0), class_labels)
            fake_images = fake_images_cache.detach()
            
            # Train diffusion
            #diffusion_loss = get_loss(model, images, t, class_labels, class_weights)
            adv_loss = get_adversarial_loss(discr, fake_images.detach(), loss_d, DEVICE)
                
            #total_loss = diffusion_loss + ADV_LOSS_WEIGHT * adv_loss 
            adv_loss.backward()
            #total_loss.backward()
            optimizer.step()
                
        if (OUTPUT_TO_FILE):
            log(f"Epoch {epoch + 1}/{FINETUNE_EPOCHS}, Adv Loss: {adv_loss.item()},")    
                    
        if ((epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), filename_diff)
            
            log('Saved checkpoint.')
            generate_and_save(f'step3_epoch{epoch + 1}')

    torch.save(model.state_dict(), filename_diff)
    log(f'Saved models.')
    
if __name__ == "__main__":
    torch.manual_seed(42)
    log(f"Using {DEVICE}")
    
    dataloader = load_data()
    class_weights = get_class_weights(dataloader)
    
    model = UNet()
    discr = Discriminator()
    
    if torch.cuda.device_count() > 1:
        log(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
        discr = nn.DataParallel(discr)
        
    model.to(DEVICE)
    discr.to(DEVICE)    
        
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_d = Adam(discr.parameters(), lr=LEARNING_RATE_DISCR)
    
    loss_d = nn.BCELoss()
    
    log('\nSTEP 1')
    log('Training diffusion model class free...')
    train_diffusion_class_free()
    #model = load_model(model, f"{MODEL_PATH}_cf.pth")
    
    log('\nSTEP 2')
    log('Training diffusion model with class embeddings...')
    train_diffusion_class()
    #model = load_model(model,f"{MODEL_PATH}_ce.pth")
    
    log('\nSTEP 3')
    log('Training discriminator...')
    train_discriminator()
    #discr = load_model(discr, f"{MODEL_PATH}_discr.pth")
    
    log('\nSTEP 4')
    log('Finetuning...')
    finetune()
    
    #model = load_model(model, f"{MODEL_PATH}_ce.pth")
    #discr = load_model(discr, f"{MODEL_PATH}_discr.pth")
    
    log('\nGenerating images...')
    
    generate_and_save('final1')
    generate_and_save('final2')
    generate_and_save('final3')
    generate_and_save('final4')
    generate_and_save('final5')
    
    log('\nDone.')
