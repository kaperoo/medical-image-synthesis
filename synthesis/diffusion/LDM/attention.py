import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Activation(nn.ReLU):
    def forward(self, x):
        return super().forward(x)
    
def normalization(channels):
    # return nn.BatchNorm2d(channels)
    return nn.GroupNorm(32, channels)

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
        self.norm = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, num_heads)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.pos_embed(x)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        x = self.norm(x)
        x, _ = self.mha(x, x, x, need_weights=False)
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
            # nn.BatchNorm2d(y_channels),
            normalization(y_channels),
            # nn.ReLU()
            # nn.SiLU()
            Activation()
        )

        self.ypre = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(y_channels, y_channels, 1),
            # nn.BatchNorm2d(y_channels),
            normalization(y_channels),
            # nn.ReLU(),
            # nn.SiLU()
            Activation()
        )

        self.xskip = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(x_channels, x_channels, 3, padding=1),
            nn.Conv2d(x_channels, y_channels, 1),
            # nn.BatchNorm2d(y_channels),
            normalization(y_channels),
            # nn.ReLU()
            # nn.SiLU()
            Activation()
        )

        self.asigm = nn.Sequential(
            nn.Conv2d(y_channels, y_channels, 1),
            # nn.BatchNorm2d(y_channels),
            normalization(y_channels),
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

# --- CBAM ATTENTION ---
# src: https://arxiv.org/abs/1807.06521
#      https://github.com/Jongchan/attention-module/

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        # self.relu = nn.ReLU() if relu else None
        # self.relu = nn.SiLU() if relu else None
        self.relu = Activation() if relu else None

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
            # nn.ReLU(),
            # nn.SiLU(),
            Activation(),
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