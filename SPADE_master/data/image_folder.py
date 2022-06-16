"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch.utils.data as data
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_rec(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


def make_dataset(dir, recursive=False, read_cache=False, write_cache=False, remove_gray_imgs=False, is_image=True):
    images = []
    gray_scale_imgs = ['11353622803_0de2b7b088_b',
                        '115758430_d061c87b5a_b',
                        '12984815745_65b85ac750_b',
                        '143192080_6f625f9395_b',
                        '143192081_c18fd910ef_b',
                        '14449828172_952d9c1ccc_b',
                        '15389204151_392009672e_b',
                        '15699264089_f051f2f5b0_b',
                        '16043835394_4512120627_b',
                        '17300502775_4887bd6968_b',
                        '17756857269_249d0baf82_b',
                        '18303741595_7110c954ef_b',
                        '190934991_4b2f916259_b',
                        '21205038811_bbe4f046a2_b',
                        '2230377496_dd602938ab_b',
                        '23427149724_2461c39798_b',
                        '2404318197_ac0494c5a0_b',
                        '24444852147_d3a866e78e_b',
                        '24717951516_3c16c0f417_b',
                        '24916631769_cf56300050_b',
                        '25343803572_f3100a110f_b',
                        '25530291740_1537abf7ef_b',
                        '27815964045_ce4a18d1a1_b',
                        '27866308613_dc1d3fb568_b',
                        '28088321880_91f66e75be_b',
                        '30028685815_2234b40f57_b',
                        '30473275945_2004171927_b',
                        '30581959281_e4b2e1365b_b',
                        '30830526070_b92ffd8dc6_b',
                        '31272725728_9f356583be_b',
                        '33711975325_28d332df24_b',
                        '34747817584_798b7a5177_b',
                        '3498123046_47b94083ec_b',
                        '3498206776_fa841c44bf_b',
                        '3769198222_e46daf27de_b',
                        '38067729362_2a54019de5_b',
                        '40359461253_2971d838f5_b',
                        '40632196493_c97ffedc9c_b',
                        '4407077267_ce8387564b_b',
                        '440820962_9117c8be51_b',
                        '4734017177_cc3364968b_b',
                        '4836260860_04386539a6_b',
                        '49552286648_a47a82e86a_b',
                        '5327756166_c4b6118948_b',
                        '5588119616_035763822d_b',
                        '6106072019_ed4f40d295_b',
                        '6204319732_47b5743e3a_b',
                        '6228744576_3c79c3075a_b',
                        '6283928494_ef867ddfe5_b',
                        '6940995391_a10f28ebb0_b',
                        '7153595509_ba346c0a33_b',
                        '7263857212_a5973c363c_b',
                        '7277064838_5d99d7f50e_b',
                        '7290338614_96a1183c38_b',
                        '7637257794_4b3c1783ef_b',
                        '7659955716_6c1b96be16_b',
                        '7755287220_f390495a31_b',
                        '7882917054_ab970d6c70_b',
                        '8155487010_889e0df0d8_b',
                        '8578161796_032fe137e7_b',
                        '9838480145_22f35818f7_b'] # 61 imgs
    gray_scale_imgs_path = []
    extension = '.jpg' if is_image else '.png'
    for each_img in gray_scale_imgs:
        gray_scale_imgs_path.append(os.path.join(dir, each_img+extension))

    if read_cache:
        possible_filelist = os.path.join(dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                images = f.read().splitlines()
                return images

    if recursive:
        make_dataset_rec(dir, images)
    else:
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

        for root, dnames, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        
        if remove_gray_imgs:
            images = [x for x in images if x not in gray_scale_imgs_path]

    if write_cache:
        filelist_cache = os.path.join(dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in images:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
