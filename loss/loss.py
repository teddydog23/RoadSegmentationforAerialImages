import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 1. BCE + Dice Loss
def bce_dice_loss(y_pred, y_true, smooth=1e-5):
    bce = F.binary_cross_entropy(y_pred, y_true)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)

    intersection = (y_pred_flat * y_true_flat).sum(1)
    dice = (2. * intersection + smooth) / (y_pred_flat.sum(1) + y_true_flat.sum(1) + smooth)
    dice_loss = 1 - dice.mean()

    return bce + dice_loss

# ✅ 2. Focal Loss
def focal_loss(y_pred, y_true, alpha=0.8, gamma=2.0):
    y_pred = y_pred.clamp(1e-7, 1 - 1e-7)
    bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

# ✅ 3. IOU + BCE Loss
def iou_bce_loss(y_pred, y_true, smooth=1e-5):
    bce = F.binary_cross_entropy(y_pred, y_true)

    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)

    intersection = (y_pred_flat * y_true_flat).sum(1)
    union = y_pred_flat.sum(1) + y_true_flat.sum(1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    iou_loss = 1 - iou.mean()

    return bce + iou_loss
