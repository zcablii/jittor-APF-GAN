"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import ntpath
import time
# import scipy.misc
from PIL import Image
import numpy as np
from utils import *
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, params):
        self.params = params

        self.web_dir = os.path.join(params['result_dir'], 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        mkdirs([self.web_dir, self.img_dir])

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):

        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)
                

        for label, image_numpy in visuals.items():
            if isinstance(image_numpy, list):
                for i in range(len(image_numpy)):
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.png' % (epoch, label, i))
                    self.save_image(image_numpy[i], img_path, create_dir=False)
            else:
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]         
                self.save_image(image_numpy, img_path)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.params['minibatch_size'] > 8
            
            t = tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    def save_image(self, image_numpy, image_path, create_dir=False, is_img = False):
        print( create_dir)
        if create_dir:
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
        if len(image_numpy.shape) == 2:
            image_numpy = np.expand_dims(image_numpy, axis=2)
        if image_numpy.shape[2] == 1:
            image_numpy = np.repeat(image_numpy, 3, 2)
        image_pil = Image.fromarray(image_numpy)

        # save to png
        if is_img:
            image_pil.save(image_path)
        else:
            image_pil.save(image_path.replace('.jpg', '.png'))


    # save image to the disk
    def inference_save_images(self, webpage, visuals, image_path):        
        visuals = self.convert_visuals_to_numpy(visuals)        
        
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.jpg' % (name))
            save_path = os.path.join(image_dir, image_name)
            if 'input_label' == label:
                self.save_image(image_numpy, save_path, create_dir=True, is_img = False)
            else:
                self.save_image(image_numpy, save_path, create_dir=True, is_img = True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

