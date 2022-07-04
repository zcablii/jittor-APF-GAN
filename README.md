# Train

注意！oasis默认的输入是256，而且不会保留图像的长宽比，例如会把一个512×384的原图reshape成256×256的正方形
- 如果想要输入的是572，crop512，需要添加 ```--large_scale```
- 如果想要保持图像的长宽比，需要添加 ```--keep_ratio```

## baseline
```makefile
oasis_baseline:
	rlaunch --charged-group=is_biometric --preemptible=no --cpu=32 --gpu=8 --memory=102400 -- \
		python -m torch.distributed.launch --nproc_per_node=8 train.py \
			--name oasis_baseline \
			--dataset_mode nori \
			--gpu_ids 0,1,2,3,4,5,6,7 \
			--batch_size 24 \
			--label_dir s3://ylf/datasets/seg/jittor-Flickr/train/gray_label.json \
			--image_dir s3://ylf/datasets/seg/jittor-Flickr/train/imgs.json \
			--continue_train
test_oasis_baseline:
	python -m torch.distributed.launch --nproc_per_node=1 test.py \
		--name oasis_baseline \
		--dataset_mode nori \
		--gpu_ids 0 \
		--batch_size 4 \
		--label_dir s3://ylf/datasets/seg/jittor-Flickr/eval/gray_label.json \
		--image_dir s3://ylf/datasets/seg/jittor-Flickr/eval/gray_label.json \
		--ckpt_iter best
```
## baseline+seg_noise
```makefile
oasis_baseline_noise:
	rlaunch --charged-group=is_biometric --preemptible=no --cpu=32 --gpu=8 --memory=102400 -- \
		python -m torch.distributed.launch --nproc_per_node=8 train.py \
			--name oasis_baseline_noise \
			--dataset_mode nori \
			--gpu_ids 0,1,2,3,4,5,6,7 \
			--batch_size 24 \
			--label_dir s3://ylf/datasets/seg/jittor-Flickr/train/gray_label.json \
			--image_dir s3://ylf/datasets/seg/jittor-Flickr/train/imgs.json \
			--use_seg_noise
test_oasis_baseline_noise:
	python -m torch.distributed.launch --nproc_per_node=1 test.py \
		--name oasis_baseline_noise \
		--dataset_mode nori \
		--gpu_ids 0 \
		--batch_size 4 \
		--label_dir s3://ylf/datasets/seg/jittor-Flickr/eval/gray_label.json \
		--image_dir s3://ylf/datasets/seg/jittor-Flickr/eval/gray_label.json \
		--ckpt_iter best \
		--use_seg_noise
```
## spade_generator(55.8 pretrained) + oasis discriminater
```makefile
oasis_pretrained_on_spade:
	rlaunch --charged-group=is_biometric --preemptible=no --cpu=32 --gpu=8 --memory=102400 -- \
		python -m torch.distributed.launch --nproc_per_node=8 train.py \
			--name oasis_pretrained_on_spade \
			--dataset_mode nori \
			--gpu_ids 0,1,2,3,4,5,6,7 \
			--batch_size 40 \
			--label_dir s3://ylf/datasets/seg/jittor-Flickr/train/gray_label.json \
			--image_dir s3://ylf/datasets/seg/jittor-Flickr/train/imgs.json \
			--generator_name spade \
			--use_seg_noise
test_oasis_pretrained_on_spade:
	python -m torch.distributed.launch --nproc_per_node=1 test.py \
		--name oasis_pretrained_on_spade \
		--dataset_mode nori \
		--gpu_ids 0 \
		--batch_size 4 \
		--label_dir s3://ylf/datasets/seg/jittor-Flickr/eval/gray_label.json \
		--image_dir s3://ylf/datasets/seg/jittor-Flickr/eval/gray_label.json \
		--generator_name spade \
		--ckpt_iter best \
		--use_seg_noise
```