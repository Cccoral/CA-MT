#!/bin/bash

## train ##
# python ./code/train.py  --dataset_name UHCS  --batch_size 8  --model unet  --labeled_proportion 1.0  --base_lr 0.001 --gpu '1' && \
# python ./code/train.py  --dataset_name UHCS  --batch_size 8  --model unet  --labeled_proportion 0.5  --base_lr 0.001 --gpu '1' && \
# python ./code/train.py  --dataset_name UHCS  --batch_size 8  --model unet  --labeled_proportion 0.25 --base_lr 0.001 --gpu '1' 

##  test  ##
# python ./code/test.py  --dataset_name UHCS   --model unet  --labeled_proportion 1.0 && \
# python ./code/test.py  --dataset_name UHCS   --model unet  --labeled_proportion 0.5 && \
# python ./code/test.py  --dataset_name UHCS   --model unet  --labeled_proportion 0.25