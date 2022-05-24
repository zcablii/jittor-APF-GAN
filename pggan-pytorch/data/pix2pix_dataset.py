"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import utils as util
import os
from data.image_folder import make_dataset
from functools import reduce
import numpy as np
import sys

class Pix2pixDataset(BaseDataset):
    def __init__(self, trans_mode = 'fixed', crop_size=512, aspect_ratio = 4.0/3.0, isTrain=True):
        super(BaseDataset, self).__init__()

        self.isTrain = isTrain
        self.trans_mode = trans_mode
        self.aspect_ratio = aspect_ratio
        self.crop_size = crop_size
        self.model_dataset_depth_offset = 3
        self.model_depth = 0
        self.alpha   = 1.0
        self.max_dataset_depth = 10
        self.scale_factor=2
        self.no_flip = False
        self.range_in = (0, 255)

        if not isTrain:
            self.no_flip = True

    def initialize(self, label_dir, image_dir):
        label_paths, image_paths, instance_paths = self.get_paths(label_dir, image_dir)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        
        label_paths = label_paths[:sys.maxsize]
        image_paths = image_paths[:sys.maxsize]
        instance_paths = instance_paths[:sys.maxsize]

       
        for path1, path2 in zip(label_paths, image_paths):
            assert self.paths_match(path1, path2), \
                "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, label_dir, image_dir):
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def create_image_from_depth(self, image, image_depth, target_depth):
        image = image.astype(np.float32)
        depthdiff = (image_depth - target_depth)
        image = reduce(lambda acc, x: acc + image[:, x[0]::(self.scale_factor**depthdiff),
                                          x[1]::(self.scale_factor**depthdiff)],
                           [(a,b) for a in range(self.scale_factor) for b in range(self.scale_factor)], 0)\
                    / (self.scale_factor ** 2)
        return np.uint8(np.clip(np.round(image), self.range_in[0], self.range_in[1]))


    def get_image_version(self, image, image_depth, target_depth):
        if image_depth == target_depth:
            return image
        return self.create_image_from_depth(image, image_depth, target_depth)

    def alpha_fade(self, image):
        c, h, w = image.shape
        t = image.reshape(c, h // 2, 2, w // 2, 2).mean((2, 4)).repeat(2, 1).repeat(2, 2)
        image = (image + (t - image) * (1 - self.alpha))
        return image


    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.crop_size, label.size)
        transform_label = get_transform(self.trans_mode, self.crop_size, self.aspect_ratio ,self.isTrain, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = 29  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')


        image = np.array(image).astype("uint8").transpose(2, 0, 1)
        image = self.get_image_version(image, self.max_dataset_depth,
                                               self.model_depth + self.model_dataset_depth_offset)
        image = self.alpha_fade(image)
        image = Image.fromarray(np.uint8(image.transpose(1, 2, 0)))


        transform_image = get_transform(None, -1, -1,self.isTrain, params)
        image_tensor = transform_image(image)

        # if using instance maps
        instance_tensor = 0

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        # self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
