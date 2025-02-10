import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) x2"""

    def __init__(self, in_channels, out_channels, kernel_size=(3, 5), padding=(1, 2)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


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
        time_embedding_dim=128,
        num_classes=7,
    ):
        super().__init__()

        self.time_embed = nn.Linear(1, time_embedding_dim)  # Embed time

        # self.time_mlp = nn.Sequential(
        #     SinusoidalPositionEmbeddings(time_embedding_dim),
        #     nn.Linear(time_embedding_dim, time_embedding_dim),
        #     nn.ReLU(),
        # )

        self.class_embed = nn.Embedding(num_classes, time_embedding_dim)  # Embed class

        self.encoder = nn.ModuleList(
            [
                DoubleConv(in_channels + time_embedding_dim, base_c),
                DoubleConv(base_c, base_c * 2),
                DoubleConv(base_c * 2, base_c * 4),
                DoubleConv(base_c * 4, base_c * 8),
            ]
        )

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.bottleneck = DoubleConv(base_c * 8, base_c * 16)

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
                DoubleConv(base_c * 16, base_c * 8),
                DoubleConv(base_c * 8, base_c * 4),
                DoubleConv(base_c * 4, base_c * 2),
                DoubleConv(base_c * 2, base_c),
            ]
        )

        self.final_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x, t, y):
        t_emb = self.time_embed(t.unsqueeze(1))
        y_emb = self.class_embed(y)  # Get class embedding from lookup table
        cond_emb = t_emb + y_emb  # Combine time and class embeddings

        cond_emb = cond_emb.view(cond_emb.shape[0], cond_emb.shape[1], 1, 1).expand(
            -1, -1, x.shape[2], x.shape[3]
        )
        x = torch.cat([x, cond_emb], dim=1)

        enc_outs = []
        for layer in self.encoder:
            x = layer(x)
            enc_outs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i, (up, dec) in enumerate(zip(self.upconv, self.decoder)):
            x = up(x)
            x = torch.cat([x, enc_outs[-(i + 1)]], dim=1)
            x = dec(x)

        return self.final_conv(x)


# # Example usage
# model = UNetConditional()
# x = torch.randn(1, 1, 208, 560)  # Grayscale image
# t = torch.tensor([0.5])  # Diffusion timestep
# y = torch.tensor([3])  # Class label (e.g., "class 3")

# output = model(x, t, y)
# print(output.shape)  # Expected output: (1, 1, 208, 560)

if __name__ == "__main__":
    # print model summary
    model = UNetConditional()
    print(model)
