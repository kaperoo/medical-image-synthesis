#import os
import torch
import torch.nn.functional as F
from torch import nn
import math
import sys
from attention import MultiHeadSelfAttention, CBAM

sys.path.append(sys.argv[0])

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
NUM_CLASSES = 7
TIME_EMBEDDING_DIM = 128

class Activation(nn.ReLU):
    def forward(self, x):
        return super().forward(x)
    
def normalization(channels):
    # return nn.BatchNorm2d(channels)
    return nn.GroupNorm(32, channels)
    
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, num_res_blocks=2):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.up = up
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                normalization(out_ch),
                Activation(),
            )
            
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                normalization(out_ch),
                Activation(),
                nn.AvgPool2d(2),
            )

        self.res_blocks = nn.ModuleList([
            ResnetBlock(out_ch, dropout=0.1)
            for _ in range(num_res_blocks)
        ])
            
        self.bnorm1 = normalization(out_ch)
        self.bnorm2 = normalization(out_ch)
        self.relu = Activation()

        self.attn = CBAM(out_ch)

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
        h = h + time_emb
        
        if self.up:
            # For upsampling: transform first, then apply ResNet blocks
            h = self.transform(h)
            for block in self.res_blocks:
                h = block(h)
            h = self.attn(h)
        else:
            # For downsampling: apply ResNet blocks first, then transform
            for block in self.res_blocks:
                h = block(h)
            # TODO: More attention?
            # h = self.attn(h)
            h = self.transform(h)
        
        return h


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        
        self.in_layers = nn.Sequential(
            normalization(in_channels),
            Activation(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            Activation(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
            
    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip_connection(x)

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
            Activation(),
        )
        self.class_embedding = nn.Embedding(num_classes, time_emb_dim)

        # Initial projection
        self.conv0 = nn.Conv2d(latent_dim, down_channels[0], 3, padding=1)

        # Downsampling blocks
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i + 1], time_emb_dim)
            for i in range(len(down_channels) - 1)
        ])
        
        self.bottleneck = nn.Sequential(
            ResnetBlock(down_channels[-1]),
            MultiHeadSelfAttention(down_channels[-1]),
            ResnetBlock(down_channels[-1])
        )

        # Upsampling blocks
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
            for i in range(len(up_channels) - 1)
        ])

        self.output = nn.Sequential(
            normalization(up_channels[-1]),
            Activation(),
            nn.Conv2d(up_channels[-1], latent_dim, 3, padding=1),
        )

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
        
        # TODO: Residual connections in bottleneck?
        z = self.bottleneck(z)

        for i, up in enumerate(self.ups):
            residual_z = residual_inputs.pop()
            if z.shape[2:] != residual_z.shape[2:]:
                z = F.interpolate(z, size=residual_z.shape[2:], mode="bilinear", align_corners=False)
            
            z = torch.cat((z, residual_z), dim=1)
            
            z = up(z, t)
        
        return self.output(z)



if __name__ == "__main__":
    model = LatentConditionalUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))


