"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import jittor as jt
from jittor import init
from jittor import nn
import os
from collections import OrderedDict
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from util.util import get_pure_ref_dics
import numpy as np
import random
from PIL import Image
jt.flags.use_cuda = 1
opt = TestOptions().parse()
if opt.USE_AMP:
    jt.flags.auto_mixed_precision_level = 5
dataloader = data.create_dataloader(opt)
model = Pix2PixModel(opt)
model.eval()
visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    generated = model(data_i, mode='inference')
    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print(('process image... %s' % img_path[b]))
        visuals = OrderedDict([('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:(b + 1)])
webpage.save()

if opt.use_pure:
    ref_img_dir = opt.train_img_ref_path
    ref_label_dir = opt.train_label_ref_path
    target =os.path.join(webpage.get_image_dir(),'synthesized_image')
    ref_dic,test_single_label_dic = get_pure_ref_dics(ref_label_dir, opt.label_dir)
    for label in test_single_label_dic.keys():
        res_list = random.sample(ref_dic[label],len(test_single_label_dic[label]))
        for img_name, ref_img_name in zip(test_single_label_dic[label], res_list):
                out_name = os.path.join(target, img_name.split('.')[0]+'.jpg')
                ref_img_name = ref_img_dir+ref_img_name
                im = Image.open(ref_img_name)
                out = im.resize((512,384), Image.BICUBIC)
                out.save(out_name)
