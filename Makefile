# 572 crop 512
spade_old_D_110_sz572:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name='spade_old_D_110_sz572' \
		--label_dir='root/train/gray_label.json' \
		--image_dir='root/train/imgs.json' \
		--norm_G spectralspadesyncbatch3x3 \
		--use_pos=True --use_pos_proj=True --use_interFeature_pos=False \
		--load_size 572 \
		--continue_train
test_spade_old_D_110_sz572:
	python SPADE_master/test.py \
		--name='spade_old_D_110_sz572' \
		--dataset_mode nori \
		--label_dir='root/eval/gray_label.json' \
		--image_dir='root/eval/gray_label.json' \
		--norm_G spectralspadesyncbatch3x3 \
		--use_pos=True --use_pos_proj=True --use_interFeature_pos=False \
		--which_epoch 120


# 572 crop 512 with mask encoding
# turn on --use_vae --encode_mask
spade_old_D_110_sz572_vae:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name='spade_old_D_110_sz572_vae' \
		--label_dir='root/train/gray_label.json' \
		--image_dir='root/train/imgs.json' \
		--norm_G spectralspadesyncbatch3x3 \
		--use_pos=True --use_pos_proj=True --use_interFeature_pos=False \
		--load_size 572 \
		--use_vae --encode_mask \
		--continue_train
test_spade_old_D_110_sz572_vae:
	python SPADE_master/test.py \
		--name='spade_old_D_110_sz572_vae' \
		--label_dir='root/eval/gray_label.json' \
		--image_dir='root/eval/gray_label.json' \
		--norm_G spectralspadesyncbatch3x3 \
		--use_pos=True --use_pos_proj=True --use_interFeature_pos=False \
		--use_vae --encode_mask \
		--which_epoch 120


