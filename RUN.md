CUDA_VISIBLE_DEVICES=0 python train.py --name='label2img' --label_dir='../data/train/gray_label' --image_dir='../data/train/imgs' --pg_strategy=1 niter=210 pg_niter=180 niter_decay=30 --num_D=4


python train_implus.py --name=label2img512_fp16_SPADE_RC572_ENC_SEGNOISE --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 1 --define_load_size 572 --use_vae --encode_mask --use_seg_noise


python train.py --name=PG_60x4_D4_SEGNOISE --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 1 --pg_strategy=1 --niter=210 --pg_niter=180 --niter_decay=30 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000

python train.py --name=FastTest_PG_10x4_D4_SEGNOISE --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 0 --pg_strategy=1 --niter=35 --pg_niter=30 --niter_decay=5 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000

python train.py --name=PG_60x4_D4_SEGNOISEK1 --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 2 --pg_strategy=1 --niter=210 --pg_niter=180 --niter_decay=30 --num_D=4 --use_seg_noise --use_seg_noise_kernel=1 --display_freq=5000 --print_freq=5000

python train.py --name=PG_60x4_D4 --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 3 --pg_strategy=1 --niter=210 --pg_niter=180 --niter_decay=30 --num_D=4 --display_freq=5000 --print_freq=5000


python train.py --name=FastTest_PG_10x4_D4_SEGNOISE_FIXBUG --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 3 --pg_strategy=1 --niter=35 --pg_niter=30 --niter_decay=5 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000


python train.py --name=PG_60x4_D4_SEGNOISE_FIXSTRIDE --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 1 --pg_strategy=1 --niter=210 --pg_niter=180 --niter_decay=30 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000

python train.py --name=PG_40x4_D4_SEGNOISE_FIXSTRIDE --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 2 --pg_strategy=1 --niter=140 --pg_niter=120 --niter_decay=20 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000

python train.py --name=PG_40x3_D3_SEGNOISE_FIXSTRIDE --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 2 --pg_strategy=1 --niter=100 --pg_niter=80 --niter_decay=20 --num_D=3 --use_seg_noise --display_freq=5000 --print_freq=5000

python train.py --name=PG_40x4_D4_SEGNOISE_FIXSTRIDE_LF5 --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 2 --pg_strategy=1 --niter=200 --pg_niter=120 --niter_decay=0 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000 --lambda_feat=5

python train.py --name=PG_40x4_D4_SEGNOISE_FIXSTRIDE_LF3 --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 4 --pg_strategy=1 --niter=200 --pg_niter=120 --niter_decay=0 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000 --lambda_feat=3

python train.py --name=PG_40x4_D4_SEGNOISE_FIXSTRIDE_RESUME140 --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 3 --pg_strategy=1 --niter=200 --pg_niter=120 --niter_decay=0 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000 --lambda_feat=5 --continue_train

python train.py --name=PG_40x4_D4_SEGNOISE_FIXSTRIDE_DISN2 --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 0 --pg_strategy=1 --niter=200 --pg_niter=120 --niter_decay=0 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000 --n_layers_D=2

python train.py --name=PG_40x4_D4_SEGNOISE_FIXSTRIDE_DISN1 --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 3 --pg_strategy=1 --niter=200 --pg_niter=120 --niter_decay=0 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000 --n_layers_D=1

python train.py --name=PG_60x4_D4_SEGNOISE_FIXSTRIDE_LF3 --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 6 --pg_strategy=1 --niter=300 --pg_niter=180 --niter_decay=0 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000 --lambda_feat=3

python train.py --name=PG_40x4_D4_SEGNOISE_FIXSTRIDE_DISN2_LF5 --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 7 --pg_strategy=1 --niter=200 --pg_niter=120 --niter_decay=0 --num_D=4 --use_seg_noise --display_freq=5000 --print_freq=5000 --n_layers_D=2 --lambda_feat=5


python train.py --name=PG_55score --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 3 --use_seg_noise --display_freq=5000 --print_freq=5000 --niter=260 --pg_niter=180 --niter_decay=20 --pg_strategy=1 --num_D=4 --diff_aug='color,crop,translation' --inception_loss --continue_train --which_epoch=180

python train.py --name=PG_55score_base --label_dir=/opt/data/common/jottor2022/train/gray_label/ --image_dir=/opt/data/common/jottor2022/train/imgs/ --batchSize 10 --gpu_ids 2 --display_freq=5000 --print_freq=5000 --niter=260 --pg_niter=180 --niter_decay=20 --pg_strategy=1 --num_D=4 --diff_aug='color,crop,translation' --inception_loss --continue_train --which_epoch=180
######################################################################################

python test.py --name FastTest_PG_10x4_D4_SEGNOISE --label_dir /opt/data/common/jottor2022/eval/gray_label --image_dir /opt/data/common/jottor2022/eval/gray_label --which_epoch 40 --results_dir /xiangli/jottor2022/submit/FastTest_PG_10x4_D4_SEGNOISE --gpu_ids 0 --pg_strategy=1 --num_D=4 --use_seg_noise

python test.py --name PG_60x4_D4_SEGNOISE --label_dir /opt/data/common/jottor2022/eval/gray_label --image_dir /opt/data/common/jottor2022/eval/gray_label --which_epoch latest --results_dir /xiangli/jottor2022/submit/PG_60x4_D4_SEGNOISE_latest --gpu_ids 0 --pg_strategy=1 --num_D=4 --use_seg_noise

python test.py --name FastTest_PG_10x4_D4_SEGNOISE_FIXBUG --label_dir /opt/data/common/jottor2022/eval/gray_label --image_dir /opt/data/common/jottor2022/eval/gray_label --which_epoch 40 --results_dir /xiangli/jottor2022/submit/FastTest_PG_10x4_D4_SEGNOISE_FIXBUG --gpu_ids 3 --pg_strategy=1 --num_D=4 --use_seg_noise

python test.py --name PG_40x4_D4_SEGNOISE_FIXSTRIDE --label_dir /opt/data/common/jottor2022/eval/gray_label --image_dir /opt/data/common/jottor2022/eval/gray_label --which_epoch latest --results_dir /xiangli/jottor2022/submit/PG_60x4_D4_SEGNOISE_157e --gpu_ids 3 --use_seg_noise


python test.py --name=PG_60x4_D4_SEGNOISE_FIXSTRIDE --label_dir=/opt/data/common/jottor2022/eval/gray_label --image_dir=/opt/data/common/jottor2022/eval/gray_label --which_epoch 240 --results_dir /xiangli/jottor2022/submit/PG_60x4_D4_SEGNOISE_FIXSTRIDE_240e --gpu_ids 1 --use_seg_noise

python test.py --name=PG_40x4_D4_SEGNOISE_FIXSTRIDE_LF5 --label_dir=/opt/data/common/jottor2022/eval/gray_label --image_dir=/opt/data/common/jottor2022/eval/gray_label --which_epoch 200 --results_dir /xiangli/jottor2022/submit/PG_40x4_D4_SEGNOISE_FIXSTRIDE_LF5_200e --gpu_ids 1 --use_seg_noise

python test.py --name=PG_55score --label_dir=/opt/data/common/jottor2022/eval/gray_label --image_dir=/opt/data/common/jottor2022/eval/gray_label --which_epoch latest --results_dir /xiangli/jottor2022/submit/PG_55score_268e --gpu_ids 2 --use_seg_noise

python test.py --name=PG_55score --label_dir=/opt/data/common/jottor2022/eval/gray_label --image_dir=/opt/data/common/jottor2022/eval/gray_label --which_epoch 280 --results_dir /xiangli/jottor2022/submit/PG_55score_280e --gpu_ids 2 --use_seg_noise

python test.py --name=PG_55score_base --label_dir=/opt/data/common/jottor2022/eval/gray_label --image_dir=/opt/data/common/jottor2022/eval/gray_label --which_epoch 280 --results_dir /xiangli/jottor2022/submit/PG_55score_base_280e --gpu_ids 2

