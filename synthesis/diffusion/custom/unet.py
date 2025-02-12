import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) x2"""

    def __init__(self, in_channels, out_channels, time_embedding_dim, kernel_size=(3, 5), padding=(1, 2)):
        super().__init__()
        #self.conv = nn.Sequential(
        #    nn.Conv2d(
        #        in_channels, out_channels, kernel_size=kernel_size, padding=padding
        #    ),
        #    nn.BatchNorm2d(out_channels),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(
        #        out_channels, out_channels, kernel_size=kernel_size, padding=padding
        #    ),
        #    nn.BatchNorm2d(out_channels),
        #    nn.ReLU(inplace=True),
        #)
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels)  # Time conditioning
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, t_emb):
        #return self.conv(x)
        h = self.norm1(self.activation(self.conv1(x)))

        time_emb = self.activation(self.time_mlp(t_emb))  # Shape: (batch, channels)
        time_emb = time_emb[:, :, None, None]  # Expand to (batch, channels, 1, 1)

        h = h + time_emb

        h = self.norm2(self.activation(self.conv2(h)))
        return h


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


class UNetConditional(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_c=64,
        time_embedding_dim=32,
        num_classes=7,
    ):
        super().__init__()

        # self.time_mlp = nn.Sequential(
        #     SinusoidalPositionEmbeddings(time_embedding_dim),
        #     nn.Linear(time_embedding_dim, time_embedding_dim),
        #     nn.ReLU(),
        # )
        # self.proj = nn.Conv2d(in_channels, base_c, kernel_size=3, padding=1)  # Project to base_c channels
        
        self.time_embed = SinusoidalPositionEmbeddings(time_embedding_dim)

        self.class_embed = nn.Embedding(num_classes, time_embedding_dim)  # Embed class

        self.encoder = nn.ModuleList(
            [
                DoubleConv(in_channels, base_c, time_embedding_dim),
                DoubleConv(base_c, base_c * 2, time_embedding_dim),
                DoubleConv(base_c * 2, base_c * 4, time_embedding_dim),
                DoubleConv(base_c * 4, base_c * 8, time_embedding_dim),
            ]
        )

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.bottleneck = DoubleConv(base_c * 8, base_c * 16, time_embedding_dim)

        self.upconv = nn.ModuleList(
            [
                nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2),
                nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2),
                nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2),
                nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2),
            ]
        )

        self.decoder = nn.ModuleList(
            [
                DoubleConv(base_c * 16, base_c * 8, time_embedding_dim),
                DoubleConv(base_c * 8, base_c * 4, time_embedding_dim),
                DoubleConv(base_c * 4, base_c * 2, time_embedding_dim),
                DoubleConv(base_c * 2, base_c, time_embedding_dim),
            ]
        )

        self.final_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x, t, y):
        #t_emb = self.time_embed(t.unsqueeze(1).float())
        t_emb = self.time_embed(t)
        y_emb = self.class_embed(y)  # Get class embedding from lookup table
        cond_emb = t_emb + y_emb  # Combine time and class embeddings

        # x = self.proj(x)

        enc_outs = []
        for layer in self.encoder:
            x = layer(x, cond_emb)
            enc_outs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, cond_emb)

        for i, (up, dec) in enumerate(zip(self.upconv, self.decoder)):
            x = up(x)
            
            if x.shape[2:] != enc_outs[-(i + 1)].shape[2:]:
                x = F.interpolate(x, size=enc_outs[-(i + 1)].shape[2:], mode="nearest")
            
            x = torch.cat([x, enc_outs[-(i + 1)]], dim=1)
            x = dec(x, cond_emb)

        return self.final_conv(x)


if __name__ == "__main__":
    # print model summary
    model = UNetConditional()
    print(model)
