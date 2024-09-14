#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:22:40 2022

@author: kui
"""

from torch import nn, einsum
from collections.abc import Iterable
import os
import copy
import torch
# 进行数据增强
from einops import rearrange

norm_func = nn.InstanceNorm3d
norm_func2d = nn.InstanceNorm2d
act_func = nn.CELU


class SSA(nn.Module):
    def __init__(self, dim, n_segment):
        super(SSA, self).__init__()
        self.scale = dim ** -0.5
        self.n_segment = n_segment

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.attend = nn.Softmax(dim=-1)
        self.to_temporal_qk = nn.Conv3d(dim, dim * 2,
                                        kernel_size=(3, 1, 1),
                                        padding=(1, 0, 0))

    def forward(self, x):
        # nw, c, d, h
        bt, c, h, w = x.shape
        t = self.n_segment
        # t = [16, 8, 4, 2][i]
        b = bt / t
        # Spatial Attention:
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)  # bt, c, h, w 在c维度进行分块
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))  # bt, hw, c
        #  -pixel attention
        # print(q.shape, k.shape)
        pixel_dots = einsum('b i c, b j c -> b i j', q, k) * self.scale
        pixel_attn = torch.softmax(pixel_dots, dim=-1)
        pixel_out = einsum('b i j, b j d -> b i d', pixel_attn, v)

        #  -channel attention
        chan_dots = einsum('b i c, b i k -> b c k', q, k) * self.scale  # c x c
        chan_attn = torch.softmax(chan_dots, dim=-1)
        chan_out = einsum('b i j, b d j -> b d i', chan_attn, v)  # hw, c

        # aggregation
        x_hat = pixel_out + chan_out
        x_hat = rearrange(x_hat, '(b t) (h w) c -> b c t h w', t=t, h=h, w=w)

        # Temporal attention
        t_qk = self.to_temporal_qk(x_hat)
        tq, tk = t_qk.chunk(2, dim=1)  # b, c, t, h, w
        tq, tk = map(lambda t: rearrange(t, 'b c t h w -> b t (c h w )'), (tq, tk))  # b, t, d
        tv = rearrange(v, '(b t) (h w) c -> b t (c h w)', t=t, h=h, w=w)  # shared value embedding
        dots = einsum('b i d, b j d -> b i j', tq, tk)  # txt
        attn = torch.softmax(dots, dim=-1)
        out = einsum('b k t, b t d -> b k d', attn, tv)  # txd
        out = rearrange(out, 'b t (c h w) -> (b t) c h w', h=h, w=w, c=c)
        return out


class SADA_Attention(nn.Module):
    def __init__(self, inchannel, n_segment):
        super(SADA_Attention, self).__init__()
        self.LF0 = SSA(inchannel, n_segment)
        self.LF1 = SSA(inchannel, n_segment)
        self.LF2 = SSA(inchannel, n_segment)
        self.fusion = nn.Sequential(
            nn.Conv3d(inchannel, 1, kernel_size=1, padding=0, bias=False),
            norm_func(1, affine=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # n 批处理 c 通道数 d 深度/切片数 w 宽度 h 高度
        n, c, d, w, h = x.size()
        # 复制本地 转置为 n, d, c, w, h

        localx = copy.copy(x).transpose(1, 2).contiguous().view(n * d, c, w, h)  # N,C,H*W
        localx = self.LF0(localx).transpose(1, 2).contiguous()
        x0 = localx.view(n, c, d, w, h)  # N,C,H*W 
        #  n, w, c, d, h
        localx = copy.copy(x).permute(0, 3, 1, 2, 4).contiguous().view(n * w, c, d, h)  # N,C,H*W
        localx = self.LF1(localx)
        # n, c, d, w, h
        x1 = localx.view(n, w, c, d, h).permute(0, 2, 3, 1, 4).contiguous()  # N,C,H*W 

        # n, h, c, d, w
        localx = copy.copy(x).permute(0, 4, 1, 2, 3).contiguous().view(n * h, c, d, w)  # N,C,H*W
        localx = self.LF2(localx)
        x2 = localx.view(n, h, c, d, w).permute(0, 2, 3, 4, 1).contiguous()  # N,C,H*W 

        return x0 + x1 + x2


class MVCSBlock(nn.Module):
    def __init__(self, inchannel, outchannel, num_heads, atten):
        super(MVCSBlock, self).__init__()
        self.atten = atten
        self.conv_0 = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=1, padding=0, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU(),
        )

        self.Atten = SADA_Attention(outchannel, num_heads)

        self.conv_1 = nn.Sequential(
            nn.Conv3d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv3d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv_0(x)
        # residual = x
        if self.atten:
            x = self.Atten(x)
        out = self.conv_1(x)
        return self.conv_2(out)  # + residual


class Blocks(nn.Module):
    def __init__(self, inchannel, outchannel, num_heads, atten=[False, False]):
        super(Blocks, self).__init__()
        self.block0 = MVCSBlock(inchannel, outchannel, num_heads, atten[0])
        self.block1 = MVCSBlock(outchannel, outchannel, num_heads, atten[1])
        self.conv_0 = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=1, padding=0, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU(),
        )
        self.DropLayer = nn.Dropout(0.2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # print(x.shape)
        residual = x
        # MVCSBlock 两个将维度变为需要的维度
        x = self.block0(x)
        x = self.DropLayer(x)
        x = self.block1(x)
        return x + self.conv_0(residual)


class Conv3dUnit(nn.Module):
    def __init__(self, inchannel, outchannel=1, kernel_size=1):
        super(Conv3dUnit, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class InputUnit(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3):
        super(InputUnit, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class MMMNet(nn.Module):

    def __init__(self,
                 inchannel=4,
                 num_classes=2,
                 num_head=[16, 8, 4, 2],
                 drop_rate=0.2,
                 num_class=3
                 ):
        super(MMMNet, self).__init__()
        base_channel = 64
        num_heads = num_head

        self.A_input = InputUnit(inchannel, base_channel)
        self.Pooling = nn.AvgPool3d(2, 2)

        self.A_conv0 = Blocks(base_channel, base_channel * 2, num_heads[0], [False, False])

        self.A_conv1 = Blocks(base_channel * 2, base_channel * 4, num_heads[1], [True, True])

        self.A_conv2 = Blocks(base_channel * 4, base_channel * 8, num_heads[2], [True, True])

        self.A_conv3 = Blocks(base_channel * 8, base_channel * 16, num_heads[3], [True, True])

        # self.up0 = nn.ConvTranspose2d(num_heads[3], num_heads[2], kernel_size=2, stride=2, )
        # self.up1 = nn.ConvTranspose2d(num_heads[2], num_heads[1], kernel_size=2, stride=2, )
        # self.up2 = nn.ConvTranspose2d(num_heads[1], num_heads[0], kernel_size=2, stride=2, )
        # self.up3 = nn.ConvTranspose2d(num_heads[0], num_heads[0] * 2, kernel_size=2, stride=2, )
        self.up0 = nn.Upsample(scale_factor=2, mode="trilinear")

        self.A_conv_d0 = Blocks(base_channel * 16, base_channel * 8, num_heads[3], [False, False])

        self.A_conv_d1 = Blocks(base_channel * 8, base_channel * 4, num_heads[2], [False, False])

        self.A_conv_d2 = Blocks(base_channel * 4, base_channel * 2, num_heads[1], [False, False])

        self.A_conv_d3 = Blocks(base_channel * 2, base_channel, num_heads[0], [False, False])
        self.ClassHead = nn.Sequential(
            nn.Conv3d(base_channel, base_channel // 2, kernel_size=3, padding=1, bias=False),
            nn.Conv3d(base_channel // 2, base_channel // 4, kernel_size=3, padding=1, bias=False),
            nn.Conv3d(base_channel // 4, num_class, kernel_size=3, padding=1, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d): \
                    m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')

    def forward(self, x):
        n, c, d, h, w = x.size()
        # InputUnit
        x0 = self.A_input(x)
        # AvgPool3d
        x0_1 = self.Pooling(x0)
        # Blocks
        x1 = self.A_conv0(x0_1)

        x2 = self.Pooling(x1)
        x2 = self.A_conv1(x2)

        x3 = self.Pooling(x2)
        x3 = self.A_conv2(x3)

        x4 = self.Pooling(x3)
        x4 = self.A_conv3(x4)

        x_d0 = self.A_conv_d0(x4)
        x_d0 = self.up0(x_d0) + x3

        x_d1 = self.A_conv_d1(x_d0)
        x_d1 = self.up0(x_d1) + x2

        x_d2 = self.A_conv_d2(x_d1)
        x_d2 = self.up0(x_d2) + x1

        x_d3 = self.A_conv_d3(x_d2)
        x_d3 = self.up0(x_d3) + x0

        # 分类线性层
        # out = self.ClassHead(nn.AdaptiveMaxPool3d((1, 1, 1))(x4).view(x4.shape[0], -1))
        out = self.ClassHead(x_d3)
        return out


# 主要模型
def MVCSModel(**kwargs):
    model = MMMNet(**kwargs)
    return model


def set_train_by_names(model, freeze_layer_names, freeze=True):
    layer_names = freeze_layer_names
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]

    for name, child in model.named_children():
        if name not in layer_names:
            for param in child.parameters():
                param.requires_grad = not freeze
        else:
            for param in child.parameters():
                param.requires_grad = freeze


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
    b = torch.zeros((1, 4, 192, 128, 128))
    # b = torch.zeros((1, 1, 32, 32, 32)).cuda()
    # net = nn.DataParallel(Model())
    net = MVCSModel()
    # net = net.cuda()
    out = net(b)
    print(out.shape)
