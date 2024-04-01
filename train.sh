#!/usr/bin/env bash


### dataset_name: realBR (output_num=9), GOPRO-VFI_copy(output_num=7)
### Please update "input_dir", "dataset_name" and etc.

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port=17533 --nproc_per_node=2 train.py \
	    --world_size=2 \
        --input_dir='/media/zhongyi/D/data' \
        --dataset_name='realBR' \
		--output_dir='./train_log' \
		--output_num=9 --data_mode1='Blur' --data_mode2='RS' --epoch=800 --batch_size=4 --batch_size_val=2 #--resume=True 


