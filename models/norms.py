import torch.nn.utils.spectral_norm as spectral_norm
from models.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn as nn
import torch.nn.functional as F
import torch


class SPADE(nn.Module):
    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.opt = opt
        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        if opt.use_seg_noise:
            k = opt.use_seg_noise_kernel
            self.seg_noise_var = nn.Conv2d(label_nc, norm_nc, k, padding=(k-1)//2)
            self.seg_noise_var.weight.data.fill_(0)
            self.seg_noise_var.bias.data.fill_(0)
            print('use seg noise var!!!! initialize all 0, use kernel:', opt.use_seg_noise_kernel)

    def forward(self, x, segmap):
        if self.opt.use_seg_noise:
            seg = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
            noise = self.seg_noise_var(seg)
            added_noise = (torch.randn(noise.shape[0], 1, noise.shape[2], noise.shape[3]).cuda() * noise)
            normalized = self.first_norm(x + added_noise)
        else:
            normalized = self.first_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


def get_spectral_norm(opt):
    if opt.no_spectral_norm:
        return torch.nn.Identity()
    else:
        return spectral_norm


def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == 'instance':
        return nn.InstanceNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'syncbatch':
        if opt.distributed:
            return nn.SyncBatchNorm(norm_nc, affine=False)
        else:
            return SynchronizedBatchNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'batch':
        return nn.BatchNorm2d(norm_nc, affine=False)
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % opt.param_free_norm)