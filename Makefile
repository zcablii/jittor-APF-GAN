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


# 572 crop 512 with mask encoding
# Specify: --use_vae --encode_mask
spade_old_D_110_sz572_enc:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name='spade_old_D_110_sz572_enc' \
		--label_dir='root/train/gray_label.json' \
		--image_dir='root/train/imgs.json' \
		--norm_G spectralspadesyncbatch3x3 \
		--use_pos=True --use_pos_proj=True --use_interFeature_pos=False \
		--load_size 572 \
		--use_vae --encode_mask \
		--continue_train


# 使用中间feature，UNET形式
# Specify: --use_intermediate --use_intermediate_type replace
spade_old_D_110_sz572_enc_replace:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name='spade_old_D_110_sz572_enc_replace' \
		--label_dir='root/train/gray_label.json' \
		--image_dir='root/train/imgs.json' \
		--norm_G spectralspadesyncbatch3x3 \
		--use_pos=True --use_pos_proj=True --use_interFeature_pos=False \
		--load_size 572 \
		--use_vae --encode_mask --use_intermediate --use_intermediate_type replace \
		--continue_train


# 使用中间feature，把feature和segmap进行元素相加
# Specify: --use_intermediate --use_intermediate_type add
spade_old_D_110_sz572_enc_add:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name='spade_old_D_110_sz572_enc_add' \
		--label_dir='root/train/gray_label.json' \
		--image_dir='root/train/imgs.json' \
		--norm_G spectralspadesyncbatch3x3 \
		--use_pos=True --use_pos_proj=True --use_interFeature_pos=False \
		--load_size 572 \
		--use_vae --encode_mask --use_intermediate --use_intermediate_type add \
		--continue_train


# 使用中间feature，sean: https://arxiv.org/pdf/1911.12861.pdf
# Specify: --use_intermediate --use_intermediate_type sean
spade_old_D_110_sz572_enc_sean:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name='spade_old_D_110_sz572_enc_sean' \
		--label_dir='root/train/gray_label.json' \
		--image_dir='root/train/imgs.json' \
		--norm_G spectralspadesyncbatch3x3 \
		--use_pos=True --use_pos_proj=True --use_interFeature_pos=False \
		--load_size 572 \
		--use_vae --encode_mask --use_intermediate --use_intermediate_type sean \
		--continue_train


