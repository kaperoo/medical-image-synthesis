#import os
import torch
import torch.nn.functional as F
from torch import nn
import math
import sys

sys.path.append(sys.argv[0])

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
NUM_CLASSES = 7
TIME_EMBEDDING_DIM = 128

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
                # nn.GroupNorm(8, out_ch), # GroupNorm instead of BatchNorm2d
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                # nn.SiLU(),
            )
            
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            # self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            # self.transform = nn.AdaptiveAvgPool2d((out_ch, out_ch))
            # self.transform = nn.AvgPool2d(2)
            self.transform = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                # nn.GroupNorm(8, out_ch), # GroupNorm instead of BatchNorm2d
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            )
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        # self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        # self.relu = nn.SiLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        # time_emb = time_emb[(...,) + (None,) * 2]
        time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
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

            z = torch.cat((z, residual_z), dim=1)
            z = up(z, t)
        
        return self.output(z)
        # return self.output(self.activation(self.norm(z)))



if __name__ == "__main__":
    model = LatentConditionalUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))


