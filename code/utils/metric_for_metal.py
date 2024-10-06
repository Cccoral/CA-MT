import numpy as np
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix

def get_confusion_metrix(preds, targets, device, num_classes):
    if num_classes == 5:
        metric = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=3).to(device)   # MetalDAM
    else:
        metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)   # UHCS
    confusion_matrix = metric(preds, targets).cpu().numpy()
    return confusion_matrix


def pixelAccuracy(confusion_matrix):
    return np.diag(confusion_matrix).sum() / confusion_matrix.sum()


def IoU(confusion_matrix):
    intersection = np.diag(confusion_matrix) 
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(
        confusion_matrix) 
    IoU = intersection / union
    return IoU


def MIoU(IoU):
    return np.nanmean(IoU)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()
        self.max=0
        

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            if self.max<val:
                self.max =val
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count




    