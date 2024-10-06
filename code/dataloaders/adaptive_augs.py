import torch
import numpy as np

# A class-driven adaptive CutMix augmentation method with the help of labeled images
class CDAC():
    def __init__(self,num_classes):
        self.num_classes=num_classes

    def get_global_image_confidence(self,pred_u_w,conf_u_w):
        # Get the global confidence of each unlabeled image in the current batch
        with torch.no_grad():
            entropy = -torch.sum(pred_u_w * torch.log(pred_u_w + 1e-10), dim=1)
            entropy /= np.log(self.num_classes)
            confidence = 1.0 - entropy
            confidence = confidence * conf_u_w
            confidence = confidence.mean(dim=[1,2]) 
            confidence = confidence.cpu().numpy()
        return confidence
    
    def image_cutmix_under_condidence_filtering(self,img_u_s, img_u_w, img_l,
                                                gt, mask_u_w, cutmix_box1, confidence):
            # Randomly select labeled images with low confidence unlabeled images for CutMix based on global confidence
            global_threshold=confidence.mean()
            indices = np.where(confidence < global_threshold)[0] 
            img_u_s_mixed,img_u_w_mixed, mask_u_w_cutmixed= img_u_s.clone(),img_u_w.clone(),mask_u_w.clone()
            if indices.size > 0:
                mix_labeled_target=gt.long().clone()
                u_rand_index = torch.randperm(img_u_s_mixed.size()[0])
                cutmix_box_expanded = cutmix_box1.unsqueeze(1).expand(img_u_w_mixed.shape)==1
                cutmix_box= cutmix_box1 == 1

                for i in indices:
                    image_mask=cutmix_box_expanded[i]==1
                    label_mask=cutmix_box[i]==1
                    img_u_s_mixed[i,image_mask] = img_l[u_rand_index[i],image_mask]
                    img_u_w_mixed[i,image_mask] = img_l[u_rand_index[i],image_mask]
                    mask_u_w_cutmixed[i, label_mask] = mix_labeled_target[u_rand_index[i],label_mask]

            return img_u_s_mixed, img_u_w_mixed, mask_u_w_cutmixed
            
