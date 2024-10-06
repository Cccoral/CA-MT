

import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloaders.metal_datasets import (ValDataset, Resize)
from utils.metric_for_metal import *
import cv2
from networks.net_factory import net_with_ema


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data', help='path of Data')
parser.add_argument('--snapshot_path', type=str, default='./pth', help='Name of Experiment')
parser.add_argument('--dataset_name', type=str, default='MetalDAM', help='Name of dataset')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--labeled_proportion', type=float, default=1.0, help='proportion of labeled data in training set')

args = parser.parse_args()

def save_predict_img(preds, origin_size, dir_path, name):

    if args.dataset_name == 'MetalDAM':
        color_map = np.array([
            (252, 156, 27), # 0 Matrix
            (176, 240, 248), # 1 Austenite
            (57, 80, 229),   # 2 Martensite/Austenite (MA) 
            (50, 148, 0),   # 3 Precipitate
            (0, 0, 0)        # 4 Defect
        ], np.uint8)
    else:
        color_map = np.array([
            (252, 156, 27),  # 0  Ferritic matrix 
            (176, 240, 248), # 1  Cementite network 
            (57, 80, 229),   # 2  Spheroidite particles 
            (50, 148, 0),    # 3  Widmanstätten laths 
        ], np.uint8)

    upper = nn.UpsamplingBilinear2d(size=origin_size)
    preds = upper(preds)
    preds = torch.argmax(preds[0], dim=0).cpu().numpy().astype(np.uint8)
    color_out = color_map[preds]
    cv2.imwrite('{}/{}.png'.format(dir_path, args.model+'-'+name), color_out)


if __name__ == "__main__":

    snapshot_path = args.snapshot_path + '/{}'.format(args.dataset_name)
    img_save_path = os.path.join(snapshot_path, "{}_pred_images_{}".format(args.dataset_name,args.labeled_proportion))

    if not os.path.isdir(img_save_path):
        os.makedirs(img_save_path)
    
    if args.dataset_name == 'UHCS':
        num_classes = 4
        patch_size = (320, 320)
    else:
        num_classes = 5
        patch_size = (320, 320)

    static_path = snapshot_path+'/{}_best_model_{}.pth'.format(args.dataset_name,args.labeled_proportion)
    model = net_with_ema(in_chns=3,class_num=num_classes)
    model.load_state_dict(torch.load(static_path), strict=False)
    model.eval()

    val_dataset = ValDataset(
        base_dir=args.root_path,
        dataset_name=args.dataset_name,
        transform=Resize(patch_size)
    )
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    confusion_metrix = 0
    for _, sampled_batch in enumerate(valloader):
        val_image_batch, val_gt_batch, val_origin_gt, img_name= sampled_batch['image'], sampled_batch['label'], sampled_batch['origin_gt'],sampled_batch['name'][0]
        val_image_batch, val_gt_batch, val_origin_gt = val_image_batch.cuda(), val_gt_batch.cuda(), val_origin_gt.cuda()
        val_output = model(val_image_batch)

        confusion_metrix += get_confusion_metrix(val_output, val_gt_batch, torch.device('cuda'), num_classes=num_classes)
        save_predict_img(val_output, (val_origin_gt.shape[1], val_origin_gt.shape[2]), img_save_path, img_name)

    acc = pixelAccuracy(confusion_metrix)
    iou = IoU(confusion_metrix)
    miou = MIoU(iou)

    with open(snapshot_path + '/{}_best_test_{}.txt'.format(args.dataset_name, args.labeled_proportion), mode='w', encoding='utf-8') as f:
        f.write("acc: %03f, miou: %03f" % (acc, miou))
        f.write("\nIoU:" + str(iou))

