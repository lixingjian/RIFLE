#!/bin/sh

#vanilla fine-tuning
CUDA_VISIBLE_DEVICES=0 python -u train.py  --data_dir=../benchmark/CUB_200_2011 --fc_reinit_times=0

#fine-tuning with RIFLE
CUDA_VISIBLE_DEVICES=1 python -u train.py  --data_dir=../benchmark/CUB_200_2011 --fc_reinit_times=1
