import glob
import random
import os
import numpy as np
import cv2
from PIL import Image
from argparse import ArgumentParser

from util import util
parser = ArgumentParser()
parser.add_argument('label_path', type=str,
                    help=('Path to the label directory'))
parser.add_argument('gray_label_path', type=str,
                    help=('Path to the converted unit8 encoding label directory'))

args = parser.parse_args()
label_path = args.label_path
gray_label_path = args.gray_label_path
labels = sorted(glob.glob(label_path + "/*.*"))
util.mkdirs(gray_label_path)

for label_path in labels:
    photo_id = os.path.split(label_path)[-1][:-4]
    img_B = Image.open(label_path)
    img_B = np.array(img_B).astype("uint8")
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
    out_path = os.path.join(gray_label_path,photo_id+'.png')
    cv2.imwrite(out_path,img_B)