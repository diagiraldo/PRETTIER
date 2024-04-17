# D. Giraldo 
# Aug 2023

# EDSR model, based on https://github.com/sanghyun-son/EDSR-PyTorch 
import torch
import math
import torch.nn as nn

def default_2Dconv(in_channels, out_channels, kernel_size):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride = 1, padding=(kernel_size//2), bias = True)

class MeanShift(nn.Conv2d):
    def __init__(self, 
                 rgb_range,
                 rgb_mean=(0.4488, 0.4371, 0.4040), 
                 rgb_std=(1.0, 1.0, 1.0), 
                 sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(self,
                 convolution,
                 n_feats,
                 kernel_size,
                 activation,
                 res_scale,
                ):
        
        super(ResBlock, self).__init__()
        
        self.body = nn.Sequential(convolution(n_feats, n_feats, kernel_size),
                                  activation,
                                  convolution(n_feats, n_feats, kernel_size))
        self.res_scale = res_scale
        
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return res + x
    
class Upsampler(nn.Sequential):
    def __init__(self,
                 scale,
                 convolution,
                 n_feats,
                ):
        m = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                m.append(convolution(n_feats, 4 * n_feats, 3))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(convolution(n_feats, 9 * n_feats, 3))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        
        super(Upsampler, self).__init__(*m)     
        

class EDSR(nn.Module):
    
    def __init__(self,
                 n_colorchannels = 3,
                 n_resblocks = 32,
                 n_feats = 256,
                 kernel_size = 3,
                 scale = 2,
                 convolution = default_2Dconv,
                 activation = nn.ReLU(inplace=True),
                 rgb_range = 1.,
                ):
        
        super(EDSR, self).__init__()
        
        self.sub_mean = MeanShift(rgb_range = rgb_range)
        self.add_mean = MeanShift(rgb_range = rgb_range, sign=1)
        
        # define head module
        m_head = [
            convolution(n_colorchannels,
                        n_feats,
                        kernel_size)
        ]
        self.head = nn.Sequential(*m_head)
        
        # define body module
        m_body = [
            ResBlock(convolution,
                     n_feats,
                     kernel_size,
                     activation,
                     res_scale = 0.1,
                    ) for _ in range(n_resblocks)
        ]
        m_body.append(convolution(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*m_body)
        
        # define tail module
        m_tail = [
            Upsampler(scale,
                      convolution,
                      n_feats),
            convolution(n_feats,
                        n_colorchannels,
                        kernel_size)
        ]
        self.tail = nn.Sequential(*m_tail)
        
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        out = self.tail(res)
        return out