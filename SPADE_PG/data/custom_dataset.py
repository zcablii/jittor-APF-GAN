"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='scale_width_and_crop')
        # load_size = 286 if is_train else 256
        parser.set_defaults(load_size=576) # 256 or 512 for diff. input size
        parser.set_defaults(crop_size=512) # 256 or 512 for diff. input size
        parser.set_defaults(aspect_ratio=4/3)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=29)
        parser.set_defaults(batchSize=16) # 32 or 10 for diff. input size
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


    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        # print(label_paths, image_paths, instance_paths)['./datasets/coco_stuff/train_label/000000580986.png', './datasets/coco_stuff/train_label/000000475177.png', './datasets/coco_stuff/train_label/000000029286.png', './datasets/coco_stuff/train_label/000000426773.png', './datasets/coco_stuff/train_label/000000350505.png', './datasets/coco_stuff/train_label/000000371376.png', './datasets/coco_stuff/train_label/000000017914.png', './datasets/coco_stuff/train_label/000000138805.png', './datasets/coco_stuff/train_label/000000203744.png', './datasets/coco_stuff/train_label/000000184101.png', './datasets/coco_stuff/train_label/000000197384.png', './datasets/coco_stuff/train_label/000000284465.png', './datasets/coco_stuff/train_label/000000500044.png'] ['./datasets/coco_stuff/train_img/000000138805.jpg', './datasets/coco_stuff/train_img/000000580986.jpg', './datasets/coco_stuff/train_img/000000500044.jpg', './datasets/coco_stuff/train_img/000000426773.jpg', './datasets/coco_stuff/train_img/000000371376.jpg', './datasets/coco_stuff/train_img/000000029286.jpg', './datasets/coco_stuff/train_img/000000284465.jpg', './datasets/coco_stuff/train_img/000000017914.jpg', './datasets/coco_stuff/train_img/000000197384.jpg', './datasets/coco_stuff/train_img/000000475177.jpg', './datasets/coco_stuff/train_img/000000350505.jpg', './datasets/coco_stuff/train_img/000000184101.jpg', './datasets/coco_stuff/train_img/000000203744.jpg'] []

        return label_paths, image_paths, instance_paths
