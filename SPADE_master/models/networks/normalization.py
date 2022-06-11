"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            if opt.distributed:
                norm_layer = nn.SyncBatchNorm(get_out_channel(layer), affine=True)
            else:
                norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_h_sz, grid_w_sz):
    grid_h = np.arange(grid_h_sz, dtype=np.float16)
    grid_w = np.arange(grid_w_sz, dtype=np.float16)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_sz, grid_w_sz])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, use_pos=False, use_pos_proj=False, fdim=None, opt=None):
        super().__init__()

        self.opt = opt
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            if opt.distributed:
                self.param_free_norm = nn.SyncBatchNorm(norm_nc, affine=False)
            else:
                self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        self.use_pos = use_pos
        self.use_pos_proj = use_pos_proj
        # if use_pos:
        #     print('use_pos true!!!!!!!!')
        #     if use_pos_proj:
        #         print('use_pos_proj true!!!!!!!!')

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        self.pos_embed = None
        if use_pos_proj:
            self.pos_proj = nn.Conv2d(nhidden, nhidden, kernel_size=1)

        pw = ks // 2
        if opt.use_intermediate:
            if opt.use_intermediate_type == 'replace':
                self.mlp_shared = nn.Sequential(
                    nn.Conv2d(fdim, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
            elif opt.use_intermediate_type == 'add':
                self.mlp_shared_fea = nn.Sequential(
                    nn.Conv2d(fdim, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
                self.mlp_shared_seg = nn.Sequential(
                    nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
            elif opt.use_intermediate_type == 'sean':       
                self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.conv_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
                self.conv_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
                self.mlp_shared_fea = nn.Sequential(
                    nn.Conv2d(fdim, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
                self.mlp_shared_seg = nn.Sequential(
                    nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
            else:
                raise NotImplementedError
        else:
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
                nn.ReLU()
            )
        if opt.add_noise:
            self.noise_var = nn.Parameter(torch.zeros(norm_nc), requires_grad=True)

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, fea=None):

        # Part 1. generate parameter-free normalized activations
        if self.opt.add_noise:
            added_noise = (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1).cuda() *self.noise_var).transpose(1, 3)
            normalized = self.param_free_norm(x + added_noise)
        else:
            normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        if self.opt.use_intermediate:
            if self.opt.use_intermediate_type == 'replace':
                actv = self.mlp_shared(fea).clone()
            elif self.opt.use_intermediate_type == 'add':
                segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
                actv = self.mlp_shared_seg(segmap).clone()

                actv_fea = self.mlp_shared_fea(fea).clone()
                actv = actv + actv_fea
            elif self.opt.use_intermediate_type == 'sean':
                segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
                actv = self.mlp_shared_seg(segmap).clone()

                actv_fea = self.mlp_shared_fea(fea).clone()
                gamma_avg = self.conv_gamma(actv_fea)
                beta_avg = self.conv_beta(actv_fea)

                gamma_alpha = torch.sigmoid(self.blending_gamma)
                beta_alpha = torch.sigmoid(self.blending_beta)
            else:
                raise NotImplementedError
        else:
            segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
            actv = self.mlp_shared(segmap).clone()

        if self.use_pos: # default with True
            if self.pos_embed is None:
                B, C, H, W = actv.size()
                pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(C, H, W)).to(actv.device)
                self.pos_embed = pos_embed.permute(0, 1).reshape(1, C, H, W)
            if self.use_pos_proj: # default with True
                actv += self.pos_proj(self.pos_embed.to(torch.float16))
            else:
                actv += self.pos_embed

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        if self.opt.use_intermediate and self.opt.use_intermediate_type == 'sean':
            gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma
            beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta
            out = normalized * (1 + gamma_final) + beta_final
        else:
            out = normalized * (1 + gamma) + beta

        return out
