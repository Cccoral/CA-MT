#!/bin/bash

## train ##
# python ./code/train.py  --dataset_name MetalDAM  --batch_size 8  --model unet  --labeled_proportion 1.0  --base_lr 0.001  --gpu '0' && \
# python ./code/train.py  --dataset_name MetalDAM  --batch_size 8  --model unet  --labeled_proportion 0.5  --base_lr 0.001  --gpu '0' && \
# python ./code/train.py  --dataset_name MetalDAM  --batch_size 8  --model unet  --labeled_proportion 0.25 --base_lr 0.001  --gpu '0' 

## test ##
# python ./code/test.py  --dataset_name MetalDAM   --model unet  --labeled_proportion 1.0 && \
# python ./code/test.py  --dataset_name MetalDAM   --model unet  --labeled_proportion 0.5 && \
# python ./code/test.py  --dataset_name MetalDAM   --model unet  --labeled_proportion 0.25