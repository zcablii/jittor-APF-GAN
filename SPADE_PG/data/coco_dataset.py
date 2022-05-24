"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CocoDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--coco_no_portraits', action='store_true')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(aspect_ratio=1) # set to 1 no concate eror
        parser.set_defaults(label_nc=182)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else opt.phase

        label_dir = os.path.join(root, '%s_label' % phase)
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        if not opt.coco_no_portraits and opt.isTrain:
            label_portrait_dir = os.path.join(root, '%s_label_portrait' % phase)
            if os.path.isdir(label_portrait_dir):
                label_portrait_paths = make_dataset(label_portrait_dir, recursive=False, read_cache=True)
                label_paths += label_portrait_paths

        image_dir = os.path.join(root, '%s_img' % phase)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if not opt.coco_no_portraits and opt.isTrain:
            image_portrait_dir = os.path.join(root, '%s_img_portrait' % phase)
            if os.path.isdir(image_portrait_dir):
                image_portrait_paths = make_dataset(image_portrait_dir, recursive=False, read_cache=True)
                image_paths += image_portrait_paths

        if not opt.no_instance:
            instance_dir = os.path.join(root, '%s_inst' % phase)
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)

            if not opt.coco_no_portraits and opt.isTrain:
                instance_portrait_dir = os.path.join(root, '%s_inst_portrait' % phase)
                if os.path.isdir(instance_portrait_dir):
                    instance_portrait_paths = make_dataset(instance_portrait_dir, recursive=False, read_cache=True)
                    instance_paths += instance_portrait_paths

        else:
            instance_paths = []
        # print(label_paths, image_paths, instance_paths)['./datasets/coco_stuff/train_label/000000580986.png', './datasets/coco_stuff/train_label/000000475177.png', './datasets/coco_stuff/train_label/000000029286.png', './datasets/coco_stuff/train_label/000000426773.png', './datasets/coco_stuff/train_label/000000350505.png', './datasets/coco_stuff/train_label/000000371376.png', './datasets/coco_stuff/train_label/000000017914.png', './datasets/coco_stuff/train_label/000000138805.png', './datasets/coco_stuff/train_label/000000203744.png', './datasets/coco_stuff/train_label/000000184101.png', './datasets/coco_stuff/train_label/000000197384.png', './datasets/coco_stuff/train_label/000000284465.png', './datasets/coco_stuff/train_label/000000500044.png'] ['./datasets/coco_stuff/train_img/000000138805.jpg', './datasets/coco_stuff/train_img/000000580986.jpg', './datasets/coco_stuff/train_img/000000500044.jpg', './datasets/coco_stuff/train_img/000000426773.jpg', './datasets/coco_stuff/train_img/000000371376.jpg', './datasets/coco_stuff/train_img/000000029286.jpg', './datasets/coco_stuff/train_img/000000284465.jpg', './datasets/coco_stuff/train_img/000000017914.jpg', './datasets/coco_stuff/train_img/000000197384.jpg', './datasets/coco_stuff/train_img/000000475177.jpg', './datasets/coco_stuff/train_img/000000350505.jpg', './datasets/coco_stuff/train_img/000000184101.jpg', './datasets/coco_stuff/train_img/000000203744.jpg'] ['./datasets/coco_stuff/train_inst/000000580986.png', './datasets/coco_stuff/train_inst/000000475177.png', './datasets/coco_stuff/train_inst/000000029286.png', './datasets/coco_stuff/train_inst/000000426773.png', './datasets/coco_stuff/train_inst/000000350505.png', './datasets/coco_stuff/train_inst/000000371376.png', './datasets/coco_stuff/train_inst/000000017914.png', './datasets/coco_stuff/train_inst/000000138805.png', './datasets/coco_stuff/train_inst/000000203744.png', './datasets/coco_stuff/train_inst/000000184101.png', './datasets/coco_stuff/train_inst/000000197384.png', './datasets/coco_stuff/train_inst/000000284465.png', './datasets/coco_stuff/train_inst/000000500044.png']
        return label_paths, image_paths, instance_paths
