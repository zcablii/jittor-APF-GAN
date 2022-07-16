# later to rename this file as train.py
import argparse
import os
parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_path', type=str, default='../../', help='enable training with an image encoder to encode mask.')
opt, unknown = parser.parse_known_args()
print(opt.input_path)
os.system('python train_phase.py --input_path=%s --batchSize=10 --niter=180 --pg_niter=180 --pg_strategy=1 --num_D=4' % opt.input_path)
os.system("python train_phase.py --input_path=%s --batchSize=5 --niter=340 --pg_niter=180 --pg_strategy=1 --num_D=4 --save_epoch_freq=5 --diff_aug='color,crop,translation' --inception_loss --use_seg_noise --continue_train --which_epoch=180" % opt.input_path)
