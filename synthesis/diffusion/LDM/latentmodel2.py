#import os
import torch
import torch.nn.functional as F
from torch import nn
import math
import sys
from attention import SpatialTransformer

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
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, trans=True):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, time_emb_dim)
        self.trans = trans
        self.up = up
        if up:
            # self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                normalization(out_ch),
                Activation(),
            )
            
        else:
            # self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                normalization(out_ch),
                Activation(),
                nn.AvgPool2d(2),
            )

        self.res_block1 = ResnetBlock(in_ch, out_ch, time_emb_dim)
        # self.res_block2 = ResnetBlock(out_ch, out_ch, time_emb_dim)
            
        # self.bnorm1 = normalization(out_ch)
        # self.bnorm2 = normalization(out_ch)
        # self.relu = Activation()

        self.attn1 = SpatialTransformer(out_ch, 8, 1, 128)
        # self.attn2 = SpatialTransformer(out_ch, 8, 1, 128)

    def forward(self, x, t, class_emb):
        # h = self.bnorm1(self.relu(self.conv1(x)))
        
        time_emb = self.time_mlp(t)
        
        h = self.res_block1(x, time_emb)
        h = self.attn1(h, class_emb)
        # h = self.res_block2(h, time_emb)
        # h = self.attn2(h, class_emb)
        
        if self.trans and not self.up:
            x = self.transform(h)
            return x, h
        else:
            return h, None


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, d_time=128):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        
        self.in_layers = nn.Sequential(
            normalization(in_channels),
            Activation(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.embedding = nn.Sequential(
            Activation(),
            nn.Linear(d_time, out_channels),
        )

        self.out_layers = nn.Sequential(
            normalization(out_channels),
            Activation(),
            nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
            
    def forward(self, x, t_emb):
        h = self.in_layers(x)

        time_emb = self.embedding(t_emb).type(h.dtype)
        time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
        h = h + time_emb

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

        # # Downsampling blocks
        # self.downs = nn.ModuleList([
        #     Block(down_channels[i], down_channels[i + 1], time_emb_dim, trans=(i != len(down_channels) - 2))
        #     for i in range(len(down_channels - 1))
        # ])
        
        # Downsampling blocks
        # self.downs = nn.ModuleList([
        #     Block(down_channels[i], down_channels[i + 1], time_emb_dim)
        #     for i in range(len(down_channels) - 1)
        # ])

        self.downs = nn.ModuleList([
            Block(down_channels[0], down_channels[0], time_emb_dim),
            Block(down_channels[0], down_channels[1], time_emb_dim),
            Block(down_channels[1], down_channels[2], time_emb_dim),
            Block(down_channels[2], down_channels[3], time_emb_dim),
            Block(down_channels[3], down_channels[4], time_emb_dim, trans=False),
        ])
        
        # self.bottleneck = nn.Sequential(
        #     ResnetBlock(down_channels[-1]),
        #     MultiHeadSelfAttention(down_channels[-1]),
        #     # SpatialTransformer(down_channels[-1], 8, 1, 128),
        #     ResnetBlock(down_channels[-1])
        # )
        self.bres1 = ResnetBlock(down_channels[-1])
        self.attn = SpatialTransformer(down_channels[-1], 8, 1, 128)
        self.bres2 = ResnetBlock(down_channels[-1])

        # Upsampling blocks
        # self.ups = nn.ModuleList([
        #     Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True, trans=(i != len(up_channels) - 2))
        #     for i in range(len(up_channels) - 1)
        # ])

        # self.ups = nn.ModuleList([
        #     Block(up_channels[i]*2, up_channels[i + 1], time_emb_dim, up=True)
        #     for i in range(len(up_channels) - 1)
        # ])
        
        self.ups = nn.ModuleList([
            Block(up_channels[0], up_channels[1], time_emb_dim, up=True),
            Block(up_channels[1]+down_channels[3], up_channels[2], time_emb_dim, up=True),
            Block(up_channels[2]+down_channels[2], up_channels[3], time_emb_dim, up=True),
            Block(up_channels[3]+down_channels[1], up_channels[4], time_emb_dim, up=True),
            Block(up_channels[4]+down_channels[0], up_channels[4], time_emb_dim, trans=False, up=True),
        ])

        self.output = nn.Sequential(
            normalization(up_channels[-1]),
            Activation(),
            nn.Conv2d(up_channels[-1], latent_dim, 3, padding=1),
        )

    def forward(self, z, timestep, class_label):
        t = self.time_mlp(timestep)
        class_emb = self.class_embedding(class_label).unsqueeze(1)
        # t = t + class_emb

        # Initial conv
        z = self.conv0(z)
        
        # UNet processing
        residual_inputs = []
        for down in self.downs:
            z, res = down(z, t, class_emb)
            residual_inputs.append(res)
        
        # TODO: Residual connections in bottleneck?
        # z = self.bottleneck(z)
        print("z shape: ", z.shape)
        z = self.bres1(z, t)
        z = self.attn(z, class_emb)
        z = self.bres2(z, t)

        residual_inputs.pop()

        for i, up in enumerate(self.ups):
            if i != 0:
                residual_z = residual_inputs.pop()
                print("z shape: ", z.shape)
                print("residual_z shape: ", residual_z.shape)
                if z.shape[2:] != residual_z.shape[2:]:
                    z = F.interpolate(z, size=residual_z.shape[2:], mode="bilinear", align_corners=False)
                
                z = torch.cat((z, residual_z), dim=1)
            
            z, _ = up(z, t, class_emb)
        
        return self.output(z)



if __name__ == "__main__":
    model = LatentConditionalUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))


