import torch

def cls_average_confidence(max_prob,hard_label,num_classes,confidence_sum,count_per_class):
    # Calculate the average confidence for each class
    smooth_factor = 0.9  # smoothing factor
    for c in range(num_classes):
        class_mask = (hard_label == c)
        confidence_sum[c] = smooth_factor * confidence_sum[c] + (1 - smooth_factor) * max_prob[class_mask].sum() # updated confidence_sum
        count_per_class[c] = smooth_factor * count_per_class[c] + (1 - smooth_factor) * class_mask.sum()    # updated count_per_class
    average_confidence = confidence_sum / (count_per_class + 1e-8)  # updated the average confidence for each class
    return average_confidence


def adjust_thresholds(cls_thresholds, average_confidence,adjustment_factor,set_threshold):
    # Adjust thresholds dynamically for each class
    new_thresholds = cls_thresholds.clone()
    high_threshold=set_threshold[0]
    low_threshold=set_threshold[1]
    for i in range(len(cls_thresholds)):
        avg_conf=average_confidence[i]
        if avg_conf > high_threshold:  
            progress = min(1, adjustment_factor)
            new_thresholds[i] = cls_thresholds[i] + progress* (avg_conf - cls_thresholds[i])
            new_thresholds[i] = torch.round(new_thresholds[i] * 10) / 10
        elif avg_conf < low_threshold:  
           new_thresholds[i] =avg_conf
    return new_thresholds

def update_pseudo_label(max_prob,hard_label,threshold,learning_status):
    with torch.no_grad():
        over_threshold_mask = max_prob > threshold
        unused =torch.tensor(-1).cuda()
        if over_threshold_mask.any():
            learning_status = torch.where(over_threshold_mask, hard_label, unused)
        return learning_status