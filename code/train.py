import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
from torch import nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.metal_datasets import (LabeledDataset, UnlabeledDataset, ValDataset, RandomGenerator, Resize)
from dataloaders.adaptive_augs import CDAC
from networks.net_factory import net_with_ema

from utils.metric_for_metal import *
from utils.generate_pseudo import *
from utils.losses import get_current_unlabel_weight
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data', help='Name of Experiment')
parser.add_argument('--dataset_name', type=str, default='MetalDAM', help='Name of dataset')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--deterministic', type=bool,  default=True, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int,  default=1221, help='random seed')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')

parser.add_argument('--labeled_proportion', type=float, default=1.0, help='proportion of labeled data in training set')
parser.add_argument('--patch_size', type=list,  default=[320, 320], help='patch size of network input [w,h]')
parser.add_argument('--class_threshold', type=list,  default=[0.8, 0.4], help='Getting global confidence threshold')

parser.add_argument('--consistency_rampup', type=float, default=2000.0, help='consistency_rampup')
parser.add_argument('--consistency', type=float, default=1.0, help='weight to balance all losses')

parser.add_argument('--num_workers', type=int, default=4, help='num_workers for training set')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda')

def get_color_map(outputs):
    outputs=outputs.cpu().numpy().astype(np.uint8)
    color_map = np.array([
            #bgr
            (27, 156, 252), # 0 Matrix
            (248, 240, 176),# 1 Austenite
            (229, 80, 57),  # 2 Martensite/Austenite (MA) 
            (0, 148, 50),   # 3 Precipitate
            (0, 0, 0)       # 4 Defect
        ], np.uint8)
    color_outputs=color_map[outputs].squeeze(0).transpose(2,0,1)
    return color_outputs

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

def train(args, snapshot_path):
    ## Setup before model training ##
    patch_size = args.patch_size
    base_lr = args.base_lr
    labeled_bs = args.batch_size // 2
    unlabeled_bs = args.batch_size - labeled_bs

    if args.dataset_name == 'UHCS':
        max_iters = 2000
        val_per_iter = 10
        num_classes = 4
        val_bs = 4
    else:
        max_iters = 2000 
        val_per_iter = 10
        num_classes = 5
        val_bs = 1

    model = net_with_ema(in_chns=3,class_num=num_classes)
    ema_model = net_with_ema(in_chns=3,class_num=num_classes,ema=True)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total number of parameters: {total_params}')

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    ## Dataset loading  ##
    labeled_dataset = LabeledDataset(
        base_dir=args.root_path,
        dataset_name=args.dataset_name,
        labeled_proportion=args.labeled_proportion,
        transform=RandomGenerator(patch_size, data_type='labeled')
    )

    unlabeled_dataset = UnlabeledDataset(
        base_dir=args.root_path,
        dataset_name=args.dataset_name,
        transform=RandomGenerator(patch_size, data_type='unlabeled')
    )

    val_dataset = ValDataset(
        base_dir=args.root_path,
        dataset_name=args.dataset_name,
        transform=Resize(patch_size)
    )

    total_img = len(labeled_dataset) + len(unlabeled_dataset)
    logging.info("Total images: {}, labeled: {}, unlabeled: {}, val: {}".format(total_img, len(labeled_dataset), len(unlabeled_dataset), len(val_dataset)))

    labeled_loader = DataLoader(labeled_dataset, batch_size=labeled_bs, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn,drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=unlabeled_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn,drop_last=True)
    valloader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=0)

    model.train()
    ema_model.train()
    optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))
    criterion_ce = nn.CrossEntropyLoss()
    criterion_u = nn.CrossEntropyLoss(reduction='none') 

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(labeled_loader)))

    best_miou = 0.0
    best_acc = 0.0
    best_iou = []
    best_iter = 0
    iter_num = 0

    total_loss  = AverageMeter()
    total_loss_l = AverageMeter()
    total_loss_u = AverageMeter()
    total_loss_val = AverageMeter()

    batch_time=AverageMeter()
    memory_usage=AverageMeter()

    max_epoch = max_iters // len(labeled_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)   

    cls_thresholds = torch.full((num_classes,), 0.5, dtype=torch.float32, device='cuda')
    confidence_sum = torch.zeros((num_classes,), dtype=torch.float32, device='cuda')
    count_per_class = torch.zeros((num_classes,), dtype=torch.float32, device='cuda')



    for epoch in iterator: 
        loader = zip(labeled_loader,unlabeled_loader)
        for _, (labeled_data, unlabeled_data) in enumerate(loader):  
            start_time = time()
            torch.cuda.reset_peak_memory_stats()

            img_l, gt= labeled_data['image'], labeled_data['label']
            cutmix_box=unlabeled_data['cutmix']
            img_u_w, img_u_s, _=unlabeled_data['image']

            img_l, gt = img_l.cuda(), gt.cuda()
            cutmix_box = cutmix_box.cuda()
            img_u_w, img_u_s =img_u_w.cuda(), img_u_s.cuda()

            with torch.no_grad():
                pred_u_w = ema_model(img_u_w)
                pred_u_w = pred_u_w.softmax(dim=1) 
                conf_u_w,mask_u_w = pred_u_w.max(dim=1) 

            ## CDAC (Adaptive augmentation for unlabeled images) ##
                alac=CDAC(num_classes)
            global_confidence=alac.get_global_image_confidence(pred_u_w,conf_u_w)
            img_u_s_mixed,img_u_w_mixed, mask_u_w_cutmixed=alac.image_cutmix_under_condidence_filtering(
                                                                        img_u_s, img_u_w, img_l,
                                                                        gt, mask_u_w, cutmix_box, global_confidence)
            
            ## CCAT  ##
            cls_confidence=cls_average_confidence(conf_u_w,mask_u_w,num_classes,confidence_sum,count_per_class)
            adjustment_factor=iter_num/max_iters
            cls_thresholds=adjust_thresholds(cls_thresholds,cls_confidence,adjustment_factor,args.class_threshold)
            batch_threshold = torch.index_select(cls_thresholds, 0, mask_u_w.view(-1))
            batch_threshold = batch_threshold.view(mask_u_w.shape) 
            indicator = conf_u_w > batch_threshold
            model.train()
            ## Training Model ##
            num_lb,num_ulb=img_l.shape[0],img_u_s_mixed.shape[0]
            image_batch = torch.cat([img_l,img_u_s_mixed], dim=0)
            outputs= model(image_batch)
            pred_l, pred_u_s =outputs.split([num_lb, num_ulb])

            ## Computing Losses ##
            loss_l = criterion_ce(pred_l, gt.long())
            loss_u_s =criterion_u(pred_u_s, mask_u_w_cutmixed) * indicator
            loss_u_s=loss_u_s.mean()
            # unlabel_weight = get_current_unlabel_weight(iter_num,args)
            unlabel_weight = get_current_unlabel_weight(epoch,args)
            loss = loss_l +  unlabel_weight * loss_u_s

            ## Updating parameters ##
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_l.update(loss_l.item())
            total_loss_u.update(loss_u_s.item())

            logging.info('iter %d : loss : %03f, loss_l: %03f,loss_u: %03f'
                         % (iter_num, loss.item(), loss_l.item(),loss_u_s.item()))

            iter_num = iter_num + 1
            update_model_ema(model,ema_model,0.99)

            end_time=time()
            elapsed=end_time-start_time
            batch_time.update(elapsed)
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage.update(peak_memory)

            ## Validating Model ##
            if iter_num > 0 and iter_num % val_per_iter == 0:
                writer.add_scalars('train', {'loss_all': loss.item()}, iter_num)
                writer.add_scalars('train', {'loss_label': loss_l.item()}, iter_num)
                writer.add_scalars('train', {'loss_unlabel': loss_u_s.item()}, iter_num)

                model.eval()

                confusion_metrix = 0
                for _, sampled_batch in enumerate(valloader):
                    val_image_batch, val_gt_batch, val_origin_gt = sampled_batch['image'], sampled_batch['label'], sampled_batch['origin_gt']
                    val_image_batch, val_gt_batch, val_origin_gt = val_image_batch.cuda(), val_gt_batch.cuda(), val_origin_gt.cuda()
                    with torch.no_grad():
                        val_output = model(val_image_batch)
                        val_loss = criterion_ce(val_output, val_gt_batch.long())
                        confusion_metrix += get_confusion_metrix(val_output, val_gt_batch, torch.device('cuda'), num_classes=num_classes)
                        total_loss_val.update(val_loss.item())

                acc = pixelAccuracy(confusion_metrix)
                iou = IoU(confusion_metrix)
                miou = MIoU(iou)

                writer.add_scalar('val_loss', val_loss.item(), iter_num)

                for class_i in range(num_classes):
                    writer.add_scalars('pseudo/threshhold',{'cls_{}_threshhold'.format(class_i+1):cls_thresholds[class_i]},iter_num)

                ### record best ###
                if miou > best_miou:
                    best_miou = miou
                    best_acc = acc
                    best_iou = iou
                    best_iter = iter_num
                    save_best_path = os.path.join(snapshot_path, '{}_best_model_{}.pth'.format(args.dataset_name,args.labeled_proportion))
                    torch.save(model.state_dict(), save_best_path)
                    with open(snapshot_path + '/best_record.txt', mode='a', encoding='utf-8') as f:
                        f.write("\n\nbest_iter_num : %d, train_seg_loss: %03f ,val_seg_loss: %03f, acc: %03f, miou: %03f" % (best_iter, total_loss.avg, total_loss_val.avg, best_acc, best_miou))
                        f.write("\nIoU:" + str(best_iou))

                logging.info('iter %d : miou : %f  acc : %f' % (iter_num, miou, acc))
                model.train()
            
            if iter_num % 20 == 0:
                image = img_l[0, 0:1, :, :]
                writer.add_image('train_label/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(pred_l, dim=1), dim=1, keepdim=True) 
                writer.add_image('train_label/Prediction', get_color_map(outputs[0, ...]), iter_num)
                labels = gt[0, ...].unsqueeze(0) 
                writer.add_image('train_label/GroundTruth', get_color_map(labels), iter_num)
            
            if iter_num >= max_iters:
                break
            
        if iter_num >= max_iters:
            iterator.close() 
            break
            
    writer.close()

    logging.info(f'batch_avg_time: {batch_time.avg}')
    logging.info(f'Average memory usage: {memory_usage.avg / (1024**2):.2f} MB')

    return "Training Finished!"

if __name__ == "__main__":
    if args.deterministic:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    proportion_folder_name = "{}-labeled".format(args.labeled_proportion)
    snapshot_path = "./result/{}/{}_{}".format(args.dataset_name, proportion_folder_name,datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # train
    train(args, snapshot_path)


