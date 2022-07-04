"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spade.base_network import BaseNetwork
from models.spade.normalization import get_nonspade_norm_layer
from models.spade.architecture import ResnetBlock as ResnetBlock
from models.spade.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.spade.encoder import ConvEncoder
import os


class SPADEGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.fc = nn.Identity()
        
        self.layer_level = 4
        if opt.num_upsampling_layers == 'more':
            self.layer_level = 5
        if opt.use_interFeature_pos:
            W = opt.crop_size // 2**(self.layer_level+1)
            H = int(W / opt.aspect_ratio)

            self.pos_emb_head = nn.Parameter(torch.zeros(1, 16 * nf, H, W), requires_grad=True).cuda()
            self.pos_emb_middle = nn.Parameter(torch.zeros(1, 16 * nf, H*2, W*2), requires_grad=True).cuda()
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        # self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        for i in range(self.layer_level):
            if opt.use_interFeature_pos:
                self.register_parameter('pos_emb_%d' % i, nn.Parameter(torch.zeros(1, int(2**(3-i) * nf), H*2**(i+2), W*2**(i+2), device="cuda"), requires_grad=True))
            if i < self.layer_level - self.opt.sr_scale:
                self.add_module('up_%d' % i, SPADEResnetBlock(int(2**(4-i) * nf), int(2**(3-i) * nf), opt))
            else:
                norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
                self.add_module('up_%d' % i, nn.Sequential(norm_layer(nn.ConvTranspose2d(int(2**(4-i) * nf), int(2**(3-i) * nf),
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)), nn.ReLU(False)))
            if i == self.layer_level - self.opt.sr_scale:
                self.res_blocks = nn.Sequential(*[ResnetBlock(int(2**(3-i) * nf),
                                  norm_layer=norm_layer,
                                  activation=nn.ReLU(False),
                                  kernel_size=opt.resnet_kernel_size) for j in range(4)])
            final_nc = int(2**(3-i) * nf)


        if opt.phase == 'train':
            self.num_mid_supervision_D = opt.num_D - 1
            if self.num_mid_supervision_D > 0 and opt.pg_niter>0:
                self.inter_conv_img = nn.ModuleList([])
                for i in range(1, self.num_mid_supervision_D+1):
                    self.inter_conv_img.append(nn.Conv2d(final_nc*(2**i), 3, 3, padding=1))
        self.out_conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        self.cur_ep = 0
        return sw, sh
    
    def pg_merge(self, low_res, high_res, alpha):
        up_res =  F.interpolate(low_res, high_res.shape[-2:])
        return high_res*alpha + up_res*(1-alpha)

    def forward(self, input, epoch=0, z=None):
        seg = input
        print_inf = False
        if self.cur_ep != epoch:
            print_inf = True
            self.cur_ep = epoch

        x = self.fc(z)

        x = self.head_0(x, seg)
        if self.opt.use_interFeature_pos: x = x + self.pos_emb_head
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        if self.opt.use_interFeature_pos: x = x + self.pos_emb_middle
        # x = self.G_middle_1(x, seg) 

        # if self.opt.num_upsampling_layers == 'more' or \
        #    self.opt.num_upsampling_layers == 'most':
        #     x = self.up(x)

        # x = self.G_middle_1(x, seg)

        results = []
        for i in range(self.layer_level):
            if i == self.layer_level - self.opt.sr_scale: 
                x = self.res_blocks(x)
            up_conv = eval(f'self.up_{i}')
            if type(up_conv) == SPADEResnetBlock:
                x = self.up(x)
            x = up_conv(x, seg)
            if self.opt.use_interFeature_pos: 
                pos_emb = eval(f'self.pos_emb_{i}')
                x = x + pos_emb
            if self.opt.phase == 'train' and self.opt.pg_strategy==2:
                if self.opt.pg_niter > 0 and self.opt.num_D - 1 > 0:
                    if epoch>=self.opt.pg_niter:
                        continue
                    lowest_D_level = epoch // (self.opt.pg_niter//(self.opt.num_D - 1))
                    if lowest_D_level - 1 + self.num_mid_supervision_D >= self.opt.num_D - 1:
                        self.num_mid_supervision_D = self.num_mid_supervision_D - 1
                        if print_inf:
                            print('del')
                        del self.inter_conv_img[self.opt.num_D - 1- lowest_D_level]
                    if self.layer_level - i - 2< self.num_mid_supervision_D and i < self.layer_level-1:
                        mid_res = self.inter_conv_img[self.layer_level - i - 2](F.leaky_relu(x, 2e-1))
                        if print_inf:
                            print('lowest_D_level: ', lowest_D_level,'inter D index: ',self.layer_level - i - 2, 'mid_res shape: ',mid_res.shape)
                        results.append(torch.tanh(mid_res))


            if self.opt.phase == 'train' and self.opt.pg_strategy in [1,3,4]:
                assert self.opt.pg_niter > 0 and self.opt.num_D - 1 > 0
               
                if epoch>=self.opt.pg_niter:
                    if hasattr(self, 'inter_conv_img'):
                        del self.inter_conv_img
                    continue
                current_level = epoch // (self.opt.pg_niter//(self.opt.num_D - 1))
                alpha = (epoch % (self.opt.pg_niter//(self.opt.num_D - 1))) / (self.opt.pg_niter//(self.opt.num_D - 1)/2) - 1
                alpha = 0 if alpha<0 else alpha
                relative_level = self.opt.num_D - current_level - 1
                
                if i == self.layer_level - relative_level - 1 and i + 1 - self.layer_level < 0:
                    mid_res = self.inter_conv_img[self.opt.num_D - current_level - 2](F.leaky_relu(x, 2e-1))
                    results.append(torch.tanh(mid_res))
                    if alpha>0:
                        # print('epoch,alpha', epoch,alpha)
                        if i+1 == self.layer_level - self.opt.sr_scale: 
                            x = self.res_blocks(x)
                        up_conv = eval(f'self.up_{i+1}')
                        if type(up_conv) == SPADEResnetBlock:
                            x = self.up(x)
                        x = up_conv(x, seg)
                        if self.opt.use_interFeature_pos: 
                            pos_emb = eval(f'self.pos_emb_{i+1}')
                            x = x + pos_emb
                        if self.opt.num_D - current_level - 3>=0:
                            mid_res = torch.tanh(self.inter_conv_img[self.opt.num_D - current_level - 3](F.leaky_relu(x, 2e-1)))
                            results[0] = self.pg_merge(results[0], mid_res, alpha)
                        else:
                            mid_res = torch.tanh(self.out_conv_img(F.leaky_relu(x, 2e-1)))
                            results[0] = self.pg_merge(results[0], mid_res, alpha)
                    break
        
        if self.opt.phase == 'train' and self.opt.pg_strategy in [1,3,4]:
            if len(results) > 0:
                return results  # list of rgb from low res to high
            else:
                x = self.out_conv_img(F.leaky_relu(x, 2e-1))
                x = torch.tanh(x)
                return x

        x = self.out_conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        if len(results)>0:
            results.append(x)
            return results
        else: 
            return x


def load_network(net, path):
    print(f'load {path}')
    weights = torch.load(path, map_location='cpu')
    net.load_state_dict(weights, strict=True)
    return net


class SPADE_Generator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.encoder = ConvEncoder(opt)
        self.generator = SPADEGenerator(opt)

        self.encoder = load_network(self.encoder, 'checkpoints/sota_55.8/280_net_E.pth')
        self.generator = load_network(self.generator, 'checkpoints/sota_55.8/280_net_G.pth')

    def forward(self, x):
        z = self.encoder(x)
        x = self.generator(x, z=z)
        return x


