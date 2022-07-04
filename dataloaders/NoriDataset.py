import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np
import refile
import json
import io
import nori2
from dataloaders.base_dataset import get_params, get_transform


def get_img_IO(nid):
    img = Image.open(io.BytesIO(nori2.Fetcher().get(nid)))
    return img

GRAY_SCALE_IMAGES = [
    '11353622803_0de2b7b088_b', '115758430_d061c87b5a_b', '12984815745_65b85ac750_b', '143192080_6f625f9395_b',
    '143192081_c18fd910ef_b', '14449828172_952d9c1ccc_b', '15389204151_392009672e_b', '15699264089_f051f2f5b0_b',
    '16043835394_4512120627_b', '17300502775_4887bd6968_b', '17756857269_249d0baf82_b', '18303741595_7110c954ef_b',
    '190934991_4b2f916259_b', '21205038811_bbe4f046a2_b', '2230377496_dd602938ab_b', '23427149724_2461c39798_b',
    '2404318197_ac0494c5a0_b', '24444852147_d3a866e78e_b', '24717951516_3c16c0f417_b', '24916631769_cf56300050_b',
    '25343803572_f3100a110f_b', '25530291740_1537abf7ef_b', '27815964045_ce4a18d1a1_b', '27866308613_dc1d3fb568_b',
    '28088321880_91f66e75be_b', '30028685815_2234b40f57_b', '30473275945_2004171927_b', '30581959281_e4b2e1365b_b',
    '30830526070_b92ffd8dc6_b', '31272725728_9f356583be_b', '33711975325_28d332df24_b', '34747817584_798b7a5177_b',
    '3498123046_47b94083ec_b', '3498206776_fa841c44bf_b', '3769198222_e46daf27de_b', '38067729362_2a54019de5_b',
    '40359461253_2971d838f5_b', '40632196493_c97ffedc9c_b', '4407077267_ce8387564b_b', '440820962_9117c8be51_b',
    '4734017177_cc3364968b_b', '4836260860_04386539a6_b', '49552286648_a47a82e86a_b', '5327756166_c4b6118948_b',
    '5588119616_035763822d_b', '6106072019_ed4f40d295_b', '6204319732_47b5743e3a_b', '6228744576_3c79c3075a_b',
    '6283928494_ef867ddfe5_b', '6940995391_a10f28ebb0_b', '7153595509_ba346c0a33_b', '7263857212_a5973c363c_b',
    '7277064838_5d99d7f50e_b', '7290338614_96a1183c38_b', '7637257794_4b3c1783ef_b', '7659955716_6c1b96be16_b',
    '7755287220_f390495a31_b', '7882917054_ab970d6c70_b', '8155487010_889e0df0d8_b', '8578161796_032fe137e7_b',
    '9838480145_22f35818f7_b'
]

class NoriDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        if opt.large_scale:
            if opt.phase == "test" or for_metrics:
                opt.load_size = 512
            else:
                opt.load_size = 572
            opt.crop_size = 512
        else:
            if opt.phase == "test" or for_metrics:
                opt.load_size = 256
            else:
                opt.load_size = 286
            opt.crop_size = 256
        opt.label_nc = 29
        opt.contain_dontcare_label = True
        opt.semantic_nc = 29 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]['path']
        image_nid = self.images[idx]['nori_id']
        image = get_img_IO(image_nid).convert('RGB')

        label_path = self.labels[idx]['path']
        label_nid = self.labels[idx]['nori_id']
        label = get_img_IO(label_nid)

        assert os.path.splitext(os.path.basename(image_path))[0] == os.path.splitext(os.path.basename(label_path))[0], '%s and %s are not matching' % (image_path, label_path)

        if self.opt.keep_ratio:
            params = get_params(self.opt, label.size)
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, color_shift=False)
            transform_image = get_transform(self.opt, params)

            image = transform_image(image)
            label = transform_label(label)
        else:
            image, label = self.transforms(image, label)

        label = label * 255
        return {"image": image, "label": label, "name": self.images[idx]['path']}

    def list_images(self):
        path_img = self.opt.image_dir
        with refile.smart_open(path_img) as f:
            images = json.loads(f.read())['info_dicts']
        path_lab = self.opt.label_dir
        with refile.smart_open(path_lab) as f:
            labels = json.loads(f.read())['info_dicts']

        images = sorted(images, key=lambda x: x['path'])
        labels = sorted(labels, key=lambda x: x['path'])

        self.ignore_list = GRAY_SCALE_IMAGES
        rm1 = []
        rm2 = []
        for path1, path2 in zip(images, labels):
            filename1_without_ext = os.path.splitext(os.path.basename(path1['path']))[0]
            filename2_without_ext = os.path.splitext(os.path.basename(path2['path']))[0]
            assert filename1_without_ext == filename2_without_ext
            if filename1_without_ext in self.ignore_list:
                rm1.append(path1)
                rm2.append(path2)
        for rm in rm1:
            images.remove(rm)
        for rm in rm2:
            labels.remove(rm)

        assert len(images) == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert os.path.splitext(os.path.basename(images[i]['path']))[0] == os.path.splitext(os.path.basename(labels[i]['path']))[0], '%s and %s are not matching' % (images[i], labels[i])

        if self.opt.phase == "train" and self.for_metrics:
            return images[:10], labels[:10], (path_img, path_lab)
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # crop
        crop_x = random.randint(0, np.maximum(0, new_width -  self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label
