"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import io
import nori2
from PIL import Image
from data.base_dataset import get_params, get_transform
from data.image_folder import make_dataset
from data.pix2pix_dataset import Pix2pixDataset
import refile
import json


def get_img_IO(nid):
    img = Image.open(io.BytesIO(nori2.Fetcher().get(nid)))
    return img


class NoriDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='scale_width_and_crop')
        if not is_train:
            parser.set_defaults(load_size=512) # 256 or 512 for diff. input size
        parser.set_defaults(crop_size=512) # 256 or 512 for diff. input size
        parser.set_defaults(aspect_ratio=4/3)
        parser.set_defaults(display_winsize=256)

        parser.add_argument('--remove_gray_imgs', action='store_false', help='ignore gray training imgs')
        parser.set_defaults(remove_gray_imgs=True) 
        parser.add_argument('--brightness', type=tuple, default=(1,1), help='training image brightness augment. Tuple of float (min, max) in range(0,inf)')
        parser.add_argument('--contrast', type=tuple, default=(1,1), help='training image contrast augment. Tuple of float (min, max) in range(0,inf)')
        parser.add_argument('--saturation', type=tuple, default=(1,1), help='training image saturation augment. Tuple of float (min, max) in range(0,inf)')
        # parser.set_defaults(brightness=(0.8,1.25))
        # parser.set_defaults(contrast=(0.8,1.25))
        # parser.set_defaults(saturation=(0.8,1.25))

        parser.set_defaults(label_nc=29)
        # parser.set_defaults(batchSize=4) # 32 or 10 for diff. input size. default to 24
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)

        parser.set_defaults(no_instance=True)
        parser.add_argument('--label_dir', type=str, default='../data/train/labels', required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, default='../data/train/imgs', required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            print('set num_upsampling_layers to more')
            parser.set_defaults(num_upsampling_layers='more')
        return parser


    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, instance_paths = self.get_paths(opt)

        # util.natural_sort(label_paths)
        label_paths = sorted(label_paths, key=lambda x: x['path'])
        # util.natural_sort(image_paths)
        image_paths = sorted(image_paths, key=lambda x: x['path'])
        # if not opt.no_instance:
        #     util.natural_sort(instance_paths)

        # label_paths = label_paths[:opt.max_dataset_size]
        # image_paths = image_paths[:opt.max_dataset_size]
        # instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1['path'], path2['path']), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size


    def get_paths(self, opt):
        label_dir = opt.label_dir
        # a list of mask paths
        # label_paths = make_dataset(label_dir, recursive=False, read_cache=True)
        with refile.smart_open(label_dir) as f:
            label_paths = json.loads(f.read())['info_dicts']

        image_dir = opt.image_dir
        # a list of image paths
        # image_paths = make_dataset(image_dir, recursive=False, read_cache=True)
        with refile.smart_open(image_dir) as f:
            image_paths = json.loads(f.read())['info_dicts']

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths


    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]['path']
        label_nid = self.label_paths[index]['nori_id']
        # label = Image.open(label_path)
        label = get_img_IO(label_nid)
        # label.save(f'images/{index}.label.png')
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]['path']
        image_nid = self.image_paths[index]['nori_id']
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        # image = Image.open(image_path)
        image = get_img_IO(image_nid)
        image = image.convert('RGB')
        # image.save(f'images/{index}.image.png')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
