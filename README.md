# CA-MT

## Introduction
This repository is for the paper:
'Confidence-Aware Mean Teacher for Semi-Supervised Metallographic Image Semantic Segmentation'. 

## Usage
1. Install 
```
pip install -r requirements.txt
```

2. Train and Test the model
```
cd ./scripts
# on MetalDAM dataset
sh MetalDAM.sh
# on UHCS dataset
sh UHCS.sh
```

The arguments are as follows:

`--dataset_name` : Name of dataset.

`--batch_size` : The batch size applied on the dataset for training the model.

`--model` : Backbone network used for training the model.

`--base_lr` : The learning rate required for training the model.

`--labeled_proportion` : Proportion of labeled data in training set.

`--gpu` : Select one for training when there are several GPUs, default 0.


