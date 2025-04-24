import torch.nn as nn

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