import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PGConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=3, stride=1, pad=1,
                 pixelnorm=True, wscale=True, act='lrelu'):
        super(PGConv2d, self).__init__()

        if wscale:
            init = lambda x: nn.init.kaiming_normal(x)
        else:
            init = lambda x: x
        self.conv = nn.Conv2d(ch_in, ch_out, ksize, stride, pad)
        init(self.conv.weight)
        if wscale:
            self.c = np.sqrt(torch.mean(self.conv.weight.data ** 2).item())
            self.conv.weight.data /= self.c
        else:
            self.c = 1.
        self.eps = 1e-8

        self.pixelnorm = pixelnorm
        if act is not None:
            self.act = nn.LeakyReLU(0.2) if act == 'lrelu' else nn.ReLU()
        else:
            self.act = None
        self.conv.cuda()

    def forward(self, x):
        h = x * self.c
        h = self.conv(h)
        if self.act is not None:
            h = self.act(h)
        if self.pixelnorm:
            mean = torch.mean(h * h, 1, keepdim=True)
            dom = torch.rsqrt(mean + self.eps)
            h = h * dom
        return h


class GFirstBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(GFirstBlock, self).__init__()
        self.c1 = PGConv2d(ch_in, ch_out, 4, 1, 3, **layer_settings)
        self.c2 = PGConv2d(ch_out, ch_out, **layer_settings)
        self.toRGB = PGConv2d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, act=None)
        # print('no elo', num_channels)

    def forward(self, x, seg, last=False):
        x = self.c1(x)
        x = self.c2(x)
        if last:
            return self.toRGB(x)
        return x


class GBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(GBlock, self).__init__()
        self.c1 = PGConv2d(ch_in, ch_out, **layer_settings)
        self.c2 = PGConv2d(ch_out, ch_out, **layer_settings)
        self.toRGB = PGConv2d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, act=None)

    def forward(self, x, seg, last=False):
        x = self.c1(x)
        x = self.c2(x)
        if last:
            x = self.toRGB(x)
        return x

import re
import torch.nn.utils.spectral_norm as spectral_norm
from sync_batchnorm import SynchronizedBatchNorm2d
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
    
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE('spadesyncbatch3x3' , fin, semantic_nc)
        self.norm_1 = SPADE('spadesyncbatch3x3', fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE('spadesyncbatch3x3', fin, semantic_nc)

        # self.toRGB = PGConv2d(fout, 3, ksize=1, pad=0, pixelnorm=False, act=None)
        self.toRGB = nn.Conv2d(fout, 3, 3, padding=1)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, last = False):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx
        if last:
            out = self.toRGB(out)
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x, seg)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class Generator(nn.Module):
    def __init__(self,
        fmap_base           = 4096,
        fmap_decay          = 1.0,
        fmap_max            = 512,
        semantic_nc         = 29,
        num_channels        = 3,
        resolution          = 512,
        normalize_latents   = True,
        wscale          = True,
        pixelnorm       = True,
        leakyrelu       = True):
        super(Generator, self).__init__()

        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 8

        def nf(stage): # num of features
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.fc = nn.Conv2d(semantic_nc, nf(1), 3, padding=1)

        # self.normalize_latents = normalize_latents
        # layer_settings = {
        #     'wscale': wscale,
        #     'pixelnorm': pixelnorm,
        #     'act': 'lrelu' if leakyrelu else 'relu'
        # }
        # self.block0 = GFirstBlock(nf(1), nf(1), num_channels, **layer_settings)
        # self.blocks = nn.ModuleList([
        #     GBlock(nf(i-1), nf(i), num_channels, **layer_settings)
        #     for i in range(2, R)
        # ])
        self.block0 = SPADEResnetBlock(nf(1), nf(1), semantic_nc)
        self.blocks = nn.ModuleList([
            SPADEResnetBlock(nf(i-1), nf(i), semantic_nc)
            for i in range(3, R)
        ])

        self.depth = 0
        self.alpha = 1.0
        self.eps = 1e-8
        self.max_depth = len(self.blocks)
        self.old_depth_alpha = (-1, -1)
        # print('model max_depth: ', self.max_depth)

    def forward(self, x):

        print_inf = False

        # if self.old_depth_alpha[0] != self.depth:
        #     # print_inf = True
        #     print('====G=====information=====')
        # if self.old_depth_alpha[1] != self.alpha:
        #     print('alpha to ', self.alpha)
        #     self.old_depth_alpha[1] = self.alpha
        if print_inf:
            print('model depth, alpha ',self.old_depth_alpha,'--->',(self.depth, self.alpha))
            print('model input: ', x.shape)
            self.old_depth_alpha = (self.depth, self.alpha)

        seg = x
        # h = x.unsqueeze(2).unsqueeze(3)
        # if self.normalize_latents:
        #     mean = torch.mean(h * h, 1, keepdim=True)
        #     dom = torch.rsqrt(mean + self.eps)
        #     h = h * dom
        h = F.interpolate(seg, size=(6, 8), mode='nearest')
        h = self.fc(h)
        h = self.block0(h, seg, self.depth == 0)
        if self.depth > 0:
            for i in range(self.depth - 1):
                h = F.upsample(h, scale_factor=2)
                h = self.blocks[i](h, seg)
            h = F.upsample(h, scale_factor=2)
            ult = self.blocks[self.depth - 1](h, seg, True)
            if self.alpha < 1.0:
                if self.depth > 1:
                    preult_rgb = self.blocks[self.depth - 2].toRGB(h)
                else:
                    preult_rgb = self.block0.toRGB(h)
            else:
                preult_rgb = 0
            h = preult_rgb * (1-self.alpha) + ult * self.alpha
        if print_inf:
            print('Generated size: ', h.shape)
            print('====--------------====')
        return h


class DBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(DBlock, self).__init__()
        self.fromRGB = PGConv2d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False)
        self.c1 = PGConv2d(ch_in, ch_in, **layer_settings)
        self.c2 = PGConv2d(ch_in, ch_out, **layer_settings)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


class DLastBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(DLastBlock, self).__init__()
        self.fromRGB = PGConv2d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False)
        self.stddev = MinibatchStddev()

        self.c1 = PGConv2d(ch_in, ch_in, **layer_settings)

        self.c2 = PGConv2d(ch_in, ch_in, **layer_settings)
        self.c3 = PGConv2d(ch_in + 1, ch_in, **layer_settings)

        self.c4 = PGConv2d(ch_in, ch_out, (3,4), 1, 0, **layer_settings)
        self.downsample = F.avg_pool2d

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.downsample(x,2)
        x = self.stddev(x)
        x = self.c3(x)
        x = self.c4(x)
        return x


def Tstdeps(val):
    return torch.sqrt(((val - val.mean())**2).mean() + 1.0e-8)


class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()
        self.eps = 1.0

    def forward(self, x):
        stddev_mean = Tstdeps(x)
        new_channel = stddev_mean.expand(x.size(0), 1, x.size(2), x.size(3))
        h = torch.cat((x, new_channel), dim=1)
        return h


class Discriminator(nn.Module):
    def __init__(self,
        # dataset_shape, # Overriden based on dataset
        fmap_base           = 4096,
        fmap_decay          = 1.0,
        fmap_max            = 512,
        wscale          = True,
        pixelnorm       = False,
        resolution = 512,
        num_channels = 3,
        leakyrelu       = True):
        super(Discriminator, self).__init__()

        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 8
        self.R = R

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        layer_settings = {
            'wscale': wscale,
            'pixelnorm': pixelnorm,
            'act': 'lrelu' if leakyrelu else 'relu'
        }
        self.blocks = nn.ModuleList([
            DBlock(nf(i), nf(i-1), num_channels, **layer_settings)
            for i in range(R-1, 2, -1)
        ] + [DLastBlock(nf(1), nf(0), num_channels, **layer_settings)])

        self.linear = nn.Linear(nf(0), 1)
        self.depth = 0
        self.alpha = 1.0
        self.eps = 1e-8
        self.max_depth = len(self.blocks) - 1
        self.old_depth_alpha = (-1, -1)
        # print('Discriminator max_depth: ', self.max_depth)

    def forward(self, x):
        print_inf = False

        # if self.old_depth_alpha[0] != self.depth:
            # print_inf = True
            # print('====D=====information=====')
        # if self.old_depth_alpha[1] != self.alpha:
        #     print('alpha to ', self.alpha)
        #     self.old_depth_alpha[1] = self.alpha
        if print_inf:
            print('model depth, alpha ',self.old_depth_alpha,'--->',(self.depth, self.alpha))
            print('model input: ', x.shape)
            self.old_depth_alpha = (self.depth, self.alpha)

        xhighres = x
        h = self.blocks[-(self.depth + 1)](xhighres, True)

        if print_inf:
            print('after ',type(self.blocks[-(self.depth + 1)]),'became shape ',h.shape)
        if self.depth > 0:
            h = F.avg_pool2d(h, 2)
            if print_inf:
                print('here h ', h.shape)
            if self.alpha < 1.0:
                xlowres = F.avg_pool2d(xhighres, 2)
                if print_inf:
                    print('alpha < 1 here xlowres ', xlowres.shape, ' use blok ', type(self.blocks[-self.depth]))
                preult_rgb = self.blocks[-self.depth].fromRGB(xlowres)
                h = h * self.alpha + (1 - self.alpha) * preult_rgb
                if print_inf:
                    print('preult_rgb ', preult_rgb.shape, ' final h ', h.shape)


        for i in range(self.depth, 0, -1):
            if print_inf:
                print('here h use blk ', type(self.blocks[-i]), '  shape ', h.shape)
            h = self.blocks[-i](h)
            if print_inf:
                print('here1  shape ', h.shape)
            if i > 1:
                h = F.avg_pool2d(h, 2)
                if print_inf:
                    print('here2  shape ', h.shape)
        if print_inf:
            print('====--------------====')
        h = self.linear(h.squeeze(-1).squeeze(-1))
        return h
