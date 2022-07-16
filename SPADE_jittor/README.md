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

## Train new model

#### PG Strategy1 pretrain (with 572 crop aug)

python train_phase.py --input_path='./data/train/' --batchSize=10 --niter=180 --pg_niter=180 --pg_strategy=1 --num_D=4

#### With above checkpoint, add inception loss, diff aug and spatial noise:

python train_phase.py --input_path='./data/train/' --batchSize=5 --niter=340 --pg_niter=180 --pg_strategy=1 --save_epoch_freq=5 --num_D=4 --diff_aug='color,crop,translation' --inception_loss --use_seg_noise --continue_train --which_epoch=180

#### Or directly run below command in one step

python train.py --input_path='./data/train/' 


## Test

#### Evaluate checkpoint FID with train set
python util/fid.py /data/train/imgs/ ./data/test/labels ./checkpoints/label2img 260 265 270 275 280 

#### Merge two checkpoints with relatively low FID value (e.g. checkpoint 265 and 280) 
python util/merge_ckpt.py ./checkpoints/label2img 265 280

#### Test merged checkpoint
python test.py --input_path='../data/test/labels' --which_epoch=avg_265_280

