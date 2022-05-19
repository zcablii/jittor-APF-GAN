import glob
import random
import os
import numpy as np

from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, mode="train", opt=None):
        super().__init__()
        self.opt = opt
        self.mode = mode
        if self.mode == 'train' or self.mode == "train_sample":
            self.files = sorted(glob.glob(os.path.join(root, mode, "imgs") + "/*.*"))
        self.labels = sorted(glob.glob(os.path.join(root, mode, "color_labels") + "/*.*"))
        self.set_attrs(total_len=len(self.labels))
        print(f"from {mode} split load {self.total_len} images.")
        
        self.mask_colors = []
        colors = [85,170,255]
        for i in colors:
            for j in colors:
                for k in colors:
                    self.mask_colors.append([i,j,k])
        self.mask_colors.append([0,85,255])
        self.mask_colors.append([0,255,85])
        self.mask_colors = np.array(self.mask_colors)


    def __getitem__(self, index):
        # img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2))
        opt = self.opt
        rand_top = random.randint(0,opt.input_hight*0.5-2)
        rand_left = random.randint(0,opt.input_width*0.5-2)
        transf = transform.Compose([
                transform.Resize(size=(int(opt.input_hight*1.5), int(opt.input_width*1.5)), mode=Image.BICUBIC),
                transform.Crop(rand_top,rand_left, opt.input_hight, opt.input_width),
                transform.ToTensor(),
                transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        # img_B = np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2)
        # for i in img_B:
        #     for j in i:
        #         ind = j[0]
        #         j[:] = self.mask_colors[ind]
        # img_B = Image.fromarray(img_B)

        if self.mode == "train" or self.mode == "train_sample":
            img_A = Image.open(self.files[index % len(self.files)])
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            img_A = transf(img_A)
        else:
            img_A = np.empty([1])
        img_B = transf(img_B)

        return img_A, img_B, photo_id, index
