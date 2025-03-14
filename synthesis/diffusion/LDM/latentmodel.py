#import os
import torch
import torch.nn.functional as F
from torch import nn
import math
import sys

sys.path.append(sys.argv[0])

from autoencoder import Encoder, Decoder  # Import the pretrained autoencoder

#TODO: middle block?

# Load pretrained encoder and decoder
encoder = Encoder().eval()  # Load trained encoder
decoder = Decoder().eval()  # Load trained decoder


#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
NUM_CLASSES = 7
TIME_EMBEDDING_DIM = 128

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            # nn.ReLU(),
            nn.SiLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.attn(x)
    
class AttentionBlock2(nn.Module):
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
    
class Attention(nn.Module):
    def __init__(self, n_channels, n_heads = 1, d_k = None, n_groups = 32):
        super().__init__()
        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x, t=None):
        _ = t
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1)
        x = x.permute(0, 2, 1)

        qkv = self.projection(x).view(batch_size, -1, self.n_heads, self.d_k * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res


class SAGAttention(nn.Module):
    def __init__(self, channels, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        query = self.query(x).view(B, C // 8, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, C // 8, H * W)
        attn = torch.bmm(query, key)
        attn = F.softmax(attn, dim=-1)

        value = self.value(x).view(B, C, H * W)
        out = torch.bmm(value, attn.permute(0, 2, 1)).view(B, C, H, W)

        mask = attn.mean(dim=1).view(B, 1, H, W).expand_as(x) > self.threshold
        blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = torch.where(mask, blurred, x)

        return out + x


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            # self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
            self.transform = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch), # GroupNorm instead of BatchNorm2d
                # nn.BatchNorm2d(out_ch),
                # nn.ReLU(),
                nn.SiLU(),
            )
            
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        # self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        # self.bnorm2 = nn.BatchNorm2d(out_ch)
        # self.relu = nn.ReLU()
        self.relu = nn.SiLU()
        # self.attn = Attention(out_ch)
        # self.attn = AttentionBlock(out_ch)
        # self.attn = AttentionBlock2(out_ch)
        # self.attn = SAGAttention(out_ch)

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Attention
        # h = self.attn(h)
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class LatentConditionalUnet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, latent_dim=4):  # Use latent_dim instead of image channels
        super().__init__()
        down_channels = (128, 256, 512, 1024, 2048)
        up_channels = (2048, 1024, 512, 256, 128)
        time_emb_dim = TIME_EMBEDDING_DIM

        # Time and class embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            # nn.SiLU(),
            nn.ReLU(),
        )
        self.class_embedding = nn.Embedding(num_classes, time_emb_dim)

        # Initial projection
        self.conv0 = nn.Conv2d(latent_dim, down_channels[0], 3, padding=1)

        # Downsampling blocks
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i + 1], time_emb_dim)
            for i in range(len(down_channels) - 1)
        ])

        # Upsampling blocks
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
            for i in range(len(up_channels) - 1)
        ])

        # self.activation = nn.SiLU()
        # self.norm = nn.GroupNorm(8, up_channels[-1])

        self.output = nn.Conv2d(up_channels[-1], latent_dim, 1)

    def forward(self, z, timestep, class_label):
        t = self.time_mlp(timestep)
        class_emb = self.class_embedding(class_label)
        t = t + class_emb

        # Initial conv
        z = self.conv0(z)
        
        # UNet processing
        residual_inputs = []
        for down in self.downs:
            z = down(z, t)
            residual_inputs.append(z)
        for up in self.ups:
            residual_z = residual_inputs.pop()
            if z.shape[2:] != residual_z.shape[2:]:
                z = F.interpolate(z, size=residual_z.shape[2:], mode="bilinear", align_corners=False)

            # Alternaive for interpolation
            # diff_h = residual_z.shape[2] - z.shape[2]
            # diff_w = residual_z.shape[3] - z.shape[3]
            # z = F.pad(z, (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2))

            z = torch.cat((z, residual_z), dim=1)
            z = up(z, t)
        
        return self.output(z)
        # return self.output(self.activation(self.norm(z)))



if __name__ == "__main__":
    model = LatentConditionalUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))


