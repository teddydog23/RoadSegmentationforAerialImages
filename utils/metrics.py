import torch
import numpy as np

def dice_score(y_pred, y_true, smooth=1e-6, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    y_true = y_true.float()
    intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
    union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def iou_score(y_pred, y_true, smooth=1e-6, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    y_true = y_true.float()
    intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
    union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def compute_miou(y_pred, y_true, smooth=1e-6, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    y_true = y_true.float()
    
    # Lớp đường (foreground)
    intersection_road = (y_pred * y_true).sum(dim=(1, 2, 3))
    union_road = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3)) - intersection_road
    iou_road = (intersection_road + smooth) / (union_road + smooth)
    
    # Lớp không phải đường (background)
    y_pred_bg = 1 - y_pred
    y_true_bg = 1 - y_true
    intersection_bg = (y_pred_bg * y_true_bg).sum(dim=(1, 2, 3))
    union_bg = y_pred_bg.sum(dim=(1, 2, 3)) + y_true_bg.sum(dim=(1, 2, 3)) - intersection_bg
    iou_bg = (intersection_bg + smooth) / (union_bg + smooth)
    
    # mIoU: trung bình IoU của hai lớp
    miou = (iou_road + iou_bg) / 2
    return miou.mean()

def accuracy_score(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    y_true = y_true.float()
    correct = (y_pred == y_true).float()
    acc = correct.mean()
    return acc.item()

def precision_score(y_pred, y_true, eps=1e-6):
    y_pred = (y_pred > 0.5).float()
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    precision = tp / (tp + fp + eps)
    return precision.item()

def recall_score(y_pred, y_true, eps=1e-6):
    y_pred = (y_pred > 0.5).float()
    tp = (y_pred * y_true).sum()
    fn = ((1 - y_pred) * y_true).sum()
    recall = tp / (tp + fn + eps)
    return recall.item()

def f1_score(y_pred, y_true, eps=1e-6):
    prec = precision_score(y_pred, y_true, eps)
    rec = recall_score(y_pred, y_true, eps)
    return 2 * prec * rec / (prec + rec + eps)

def compute_metrics(y_preds, y_trues, threshold=0.5):
    with torch.no_grad():
        dice = dice_score(y_preds, y_trues)
        iou = iou_score(y_preds, y_trues)
        miou = compute_miou(y_preds, y_trues, threshold=threshold)
    return dice, iou, miou