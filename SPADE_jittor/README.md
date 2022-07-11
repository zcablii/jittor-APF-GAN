# Semantic Image Synthesis with PG-SPADE (Jittor implementation)

![1656585143029](image/README/1656585143029.png)

## Installation

Clone this repo.

```bash
git clone https://github.com/zcablii/jittor-Torile-GanVit.git
cd jittor-Torile-GanVit-main/
```

This code requires python 3+ and Jittor 1.3. Please install dependencies by

```bash
sudo apt install python3.7-dev libomp-dev  
python3.7 -m pip install jittor  
pip install -r requirements.txt
```

## Dataset Preparation

Convert official provided dataset labels into unit8 encoding by

```bash
python preprocess_label.py path_to_label path_to_converted_label
```

## Train new model

#### PG Strategy1 pretrain (with 572 crop aug)

CUDA_VISIBLE_DEVICES=0 python train.py --name='label2img' --label_dir='../data/train/gray_label' --image_dir='../data/train/imgs' --remove_img_txt_path='./remove_130_imgs.txt' --niter=180--pg_niter=180 --pg_strategy=1 --num_D=4

#### With above checkpoint, add inception loss, diff aug and spatial noise:

CUDA_VISIBLE_DEVICES=0 python train.py --name='label2img' --label_dir='../data/train/gray_label' --image_dir='../data/train/imgs' --remove_img_txt_path='./remove_130_imgs.txt' --niter=340 --pg_niter=180 --pg_strategy=1 --num_D=4 --diff_aug='color,crop,translation' --inception_loss --use_seg_noise --continue_train --which_epoch=180

## Test

CUDA_VISIBLE_DEVICES=0 python test.py --name='label2img' --label_dir='../data/eval/gray_label' --use_seg_noise --which_epoch=340

###### Test with pure label replacement

CUDA_VISIBLE_DEVICES=0 python test.py --name='label2imgpretrain_280_finetun' --label_dir='../../CGAN/data/test/gray_label' --use_seg_noise --use_pure --train_img_ref_path='../../CGAN/data/train/imgs/' --train_label_ref_path='../../CGAN/data/train/labels/' --which_epoch=340
