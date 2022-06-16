# jittor-Torile-GanVit

```makefile
# ***************************************************************************************** #
# update !!!
# ***************************************************************************************** #
# pg
# Specify: --pg_strategy 1 --niter 210 --pg_niter 180 --niter_decay 30 --num_D 4
spade_enc_pg:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name spade_enc_pg \
		--label_dir root/train/gray_label.json \
		--image_dir root/train/imgs.json \
		--norm_G spectralspadesyncbatch3x3 \
		--load_size 572 \
		--continue_train \
		--use_vae --encode_mask \
		--pg_strategy 1 --niter 210 --pg_niter 180 --niter_decay 30 --num_D 4


# ***************************************************************************************** #
# historical command
# ***************************************************************************************** #
# encoder + inception feature loss
# Specify: --inception_loss
spade_enc_inception:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name spade_enc_inception \
		--label_dir root/train/gray_label.json \
		--image_dir root/train/imgs.json \
		--norm_G spectralspadesyncbatch3x3 \
		--load_size 572 \
		--continue_train \
		--use_vae --encode_mask \
		--inception_loss


# 使用中间feature，sean: https://arxiv.org/pdf/1911.12861.pdf
# Specify: --use_intermediate --use_intermediate_type sean
spade_enc_sean:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name spade_enc_sean \
		--label_dir root/train/gray_label.json \
		--image_dir root/train/imgs.json \
		--norm_G spectralspadesyncbatch3x3 \
		--load_size 572 \
		--continue_train \
		--use_vae --encode_mask \
		--use_intermediate --use_intermediate_type sean



# 使用中间feature，把feature和segmap进行元素相加
# Specify: --use_intermediate --use_intermediate_type add
spade_enc_add:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name spade_enc_add \
		--label_dir root/train/gray_label.json \
		--image_dir root/train/imgs.json \
		--norm_G spectralspadesyncbatch3x3 \
		--load_size 572 \
		--continue_train \
		--use_vae --encode_mask \
		--use_intermediate --use_intermediate_type add


# 使用中间feature，UNET形式
# Specify: --use_intermediate --use_intermediate_type replace
spade_enc_replace:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name spade_enc_replace \
		--label_dir root/train/gray_label.json \
		--image_dir root/train/imgs.json \
		--norm_G spectralspadesyncbatch3x3 \
		--load_size 572 \
		--continue_train \
		--use_vae --encode_mask \
		--use_intermediate --use_intermediate_type replace


# 572 crop 512 with mask encoding
# Specify: --use_vae --encode_mask
spade_enc:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name spade_enc \
		--label_dir root/train/gray_label.json \
		--image_dir root/train/imgs.json \
		--norm_G spectralspadesyncbatch3x3 \
		--load_size 572 \
		--continue_train \
		--use_vae --encode_mask


# 572 crop 512
spade:
	python -m torch.distributed.launch --nproc_per_node=8 SPADE_master/train.py \
		--name spade \
		--label_dir root/train/gray_label.json \
		--image_dir root/train/imgs.json \
		--norm_G spectralspadesyncbatch3x3 \
		--load_size 572 \
		--continue_train
```