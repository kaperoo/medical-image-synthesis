import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)        
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.pos_embed = PositionalEncodingPermute2D(channels)
        self.mha = nn.MultiheadAttention(channels, num_heads)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.pos_embed(x)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        x, _ = self.mha(x, x, x)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, x_channels, y_channels, num_heads=8):
        super(MultiHeadCrossAttention, self).__init__()
        
        # channels is # of channels in the skip connection
        self.pos_embed_x = PositionalEncodingPermute2D(x_channels)
        self.pos_embed_y = PositionalEncodingPermute2D(y_channels)
        self.mha = nn.MultiheadAttention(y_channels, num_heads)

        self.xpre = nn.Sequential(
            nn.Conv2d(x_channels, y_channels, 1),
            nn.BatchNorm2d(y_channels),
            nn.ReLU()
        )

        self.ypre = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(y_channels, y_channels, 1),
            nn.BatchNorm2d(y_channels),
            nn.ReLU(),
        )

        self.xskip = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(x_channels, x_channels, 3, padding=1),
            nn.Conv2d(x_channels, y_channels, 1),
            nn.BatchNorm2d(y_channels),
            nn.ReLU()
        )

        self.asigm = nn.Sequential(
            nn.Conv2d(y_channels, y_channels, 1),
            nn.BatchNorm2d(y_channels),
            nn.Sigmoid(),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
    
    def forward(self, x, y):

        xb, xc, xh, xw = x.shape
        yb, yc, yh, yw = y.shape

        x = x + self.pos_embed_x(x)
        y = y + self.pos_embed_y(y)
        
        xa = self.xpre(x).reshape(xb, yc, xh * xw).permute(0, 2, 1)
        ya = self.ypre(y).reshape(yb, yc, yh * yw).permute(0, 2, 1)

        z, _ = self.mha(xa, xa, ya)
        z = z.permute(0, 2, 1).reshape(xb, yc, xh, xw)
        z = self.asigm(z)

        x = self.xskip(x)
        
        # hadamard product of z and y
        y = y * z

        
        x = torch.cat((x, y), dim=1)        

        return x