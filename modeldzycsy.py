# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Date        :2021/04/14 16:05
@Author      :Wentong Liao, Kai Hu
@Email       :liao@tnt.uni-hannover.de
@Version     :0.1
@Description : Implementation of SSA-GAN
'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from sync_batchnorm import SynchronizedBatchNorm2d

BatchNorm = SynchronizedBatchNorm2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]
        x = self.sample_pooling(x).flatten(2).transpose(1, 2) # N 8x8 C
        x = self.ln(x)
        x = self.act(x)

        # x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, up_stride):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x = x.transpose(1, 2)
        x_r = x[..., 1:].reshape(B, C, 8, 8)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        return F.interpolate(x_r, size=(8 * up_stride, 8 * up_stride))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """

    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                   groups=groups)

        # if last_fusion:
        #     self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True,
        #                                   groups=groups)
        # else:
        #     self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)
        self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)
        # print('4444', x_t.shape)
        # x_st: 6 64 512 ; x_t: 6 65 512
        x_t = torch.cat((x_t[..., 0], x_st + x_t[..., 1:]), dim=-1)
        x_t = self.trans_block(x_t)
        # print('5555', x_t.shape)
        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t

class NetG(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., ngf=64, nz=100):
        super(NetG, self).__init__()

        # self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # assert depth % 3 == 0

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.fc = nn.Linear(nz, 128 * 8 * 8)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(256, embed_dim)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = 256
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=256, outplanes=stage_1_channel, res_conv=True, stride=1)
        # self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )


        # 2~3 stage 32x32
        init_stage = 2
        fin_stage = 4
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, False, 1,
                                dw_stride=4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_2_channel = 512
        # 4~6 stage; 4,5 64x64; 6 128x128
        init_stage = fin_stage  # 5
        fin_stage = 7  # 9
        for i in range(init_stage, fin_stage):
            stride = 16 if i == fin_stage-1 else 8
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_2_channel, res_conv, 1,
                                dw_stride=stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_3_channel = 256
        # 7~8 stage
        init_stage = fin_stage  # 9
        fin_stage = 9  # 13
        for i in range(init_stage, fin_stage):
            stride = 32 if i == fin_stage-1 else 16
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            out_channel = stage_3_channel if i == init_stage else stage_3_channel // 2
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, 1,
                                dw_stride=stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, last_fusion=last_fusion
                            )
                            )

        self.ngf = ngf
        self.fin_stage = fin_stage

        self.conv_img = nn.Sequential(
            BatchNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 64, 3, 1, 1),
            BatchNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )

        self.mlp = nn.Linear(512,256)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x, c):
        x = self.fc(x)  # 128 * 8 * 8
        x = x.view(x.size(0), 2 * self.ngf, 8, 8) # x (bz,128,8,8)
        x = self.conv1(x)
        c = torch.cat((c.unsqueeze(dim=1), x.view(x.size(0), x.size(1), -1).permute(0,2,1)), dim=1) #c(bz,1,256) + x(bz,64,256) = c(bz,65,256)
        x_t = self.linear1(c)
        # print('2222', x_t.shape)
        # interpolate in 1,2,4,6,8 stage
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_1(x, return_x_2=False)
        x_t = self.trans_1(x_t)
        # print('3333', x_t.shape)
        # 2 ~ final
        for i in range(2, self.fin_stage):
            if i % 2 == 0:
               x = F.interpolate(x, scale_factor=2)
            # print('start conv_trans_', i, '......')
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)
            # print('finish conv_trans_', i, '......')

        x_t = self.mlp(x_t).permute(0,2,1)[1:]
        out = x + F.interpolate(x_t, size=(x.shape[-2], x.shape[-1]))

        out = self.conv_img(out)

        # return out, fusion_mask
        # return out, [stage_mask_4, stage_mask_8, stage_mask_16, stage_mask_32,
        #              stage_mask_64, stage_mask_128, stage_mask_256]
        return out, []


class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()

        self.batch_norm2d = BatchNorm(num_features, affine=False)

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.zeros_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None, fusion_mask=None):
        x = self.batch_norm2d(x)
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        weight = weight * fusion_mask + 1
        bias = bias * fusion_mask
        return weight * x + bias


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16 + 256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):

        y = y.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)  # 128
        self.block0 = resD(ndf * 1, ndf * 2)  # 64
        self.block1 = resD(ndf * 2, ndf * 4)  # 32
        self.block2 = resD(ndf * 4, ndf * 8)  # 16
        self.block3 = resD(ndf * 8, ndf * 16)  # 8
        self.block4 = resD(ndf * 16, ndf * 16)  # 4
        self.block5 = resD(ndf * 16, ndf * 16)  # 4

        self.COND_DNET = D_GET_LOGITS(ndf)

    def forward(self, x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        return out


class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x) + self.gamma * self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True, spectral_norm=False):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


def linear(in_feat, out_feat, bias=True, spectral_norm=False):
    lin = nn.Linear(in_feat, out_feat, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(lin)
    else:
        return lin

# class G_Block(nn.Module):
#
#     def __init__(self, in_ch, out_ch, num_w=256, predict_mask=True):
#         super(G_Block, self).__init__()
#
#         self.learnable_sc = in_ch != out_ch
#         self.predict_mask = predict_mask
#         self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
#         self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
#         self.affine0 = affine(in_ch)
#         #self.affine1 = affine(in_ch)
#         self.affine2 = affine(out_ch)
#         #self.affine3 = affine(out_ch)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         if self.learnable_sc:
#             self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
#
#         if self.predict_mask:
#             self.conv_mask = nn.Sequential(nn.Conv2d(out_ch, 100, 3, 1, 1),
#                                            BatchNorm(100),
#                                            nn.ReLU(),
#                                            nn.Conv2d(100, 1, 1, 1, 0))
#
#     def forward(self, x, y=None, fusion_mask=None):
#         out = self.shortcut(x) + self.gamma * self.residual(x, y, fusion_mask)
#
#         if self.predict_mask:
#             mask = self.conv_mask(out)
#         else:
#             mask = None
#
#         return out, mask
#
#     def shortcut(self, x):
#         if self.learnable_sc:
#             x = self.c_sc(x)
#         return x
#
#     def residual(self, x, y=None, fusion_mask=None):
#         h = self.affine0(x, y, fusion_mask)
#         h = nn.ReLU(inplace=True)(h)
#         h = self.c1(h)
#
#         h = self.affine2(h, y, fusion_mask)
#         h = nn.ReLU(inplace=True)(h)
#         return self.c2(h)


# class affine(nn.Module):
#
#     def __init__(self, num_features):
#         super(affine, self).__init__()
#
#         self.batch_norm2d = BatchNorm(num_features, affine=False)
#
#         self.fc_gamma = nn.Sequential(OrderedDict([
#             ('linear1', nn.Linear(256, 256)),
#             ('relu1', nn.ReLU(inplace=True)),
#             ('linear2', nn.Linear(256, num_features)),
#         ]))
#         self.fc_beta = nn.Sequential(OrderedDict([
#             ('linear1', nn.Linear(256, 256)),
#             ('relu1', nn.ReLU(inplace=True)),
#             ('linear2', nn.Linear(256, num_features)),
#         ]))
#         self._initialize()
#
#     def _initialize(self):
#         nn.init.zeros_(self.fc_gamma.linear2.weight.data)
#         nn.init.zeros_(self.fc_gamma.linear2.bias.data)
#         nn.init.zeros_(self.fc_beta.linear2.weight.data)
#         nn.init.zeros_(self.fc_beta.linear2.bias.data)
#
#     def forward(self, x, y=None, fusion_mask=None):
#         x = self.batch_norm2d(x)
#         weight = self.fc_gamma(y)
#         bias = self.fc_beta(y)
#
#         if weight.dim() == 1:
#             weight = weight.unsqueeze(0)
#         if bias.dim() == 1:
#             bias = bias.unsqueeze(0)
#
#         size = x.size()
#         weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
#         bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
#         weight = weight * fusion_mask + 1
#         bias = bias * fusion_mask
#         return weight * x + bias