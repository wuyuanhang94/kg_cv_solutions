import torch
import torch.nn as nn

def dice_ceff(y_pred, y_gt):
    smooth = 1
    probs = torch.sigmoid(y_pred)
    intersection = torch.dot(probs.view(-1), y_gt.view(-1))
    score = (2.0 * intersection + smooth) / (probs.sum() + y_gt.sum() + smooth)
    return score

def dice_loss(y_pred, y_gt):
    return 1 - dice_ceff(y_pred, y_gt)

def bce_dice_loss(y_pred, y_gt):
    bceL = nn.BCEWithLogitsLoss()(y_pred.view(-1), y_gt.view(-1))
    diceL = dice_loss(y_pred, y_gt)
    return bceL + diceL
