import torch
import torch.nn as nn

def dice_ceff(y_pred, y_gt):
    smooth = 1
    probs = torch.sigmoid(y_pred)
    num = y_gt.size(0)
    predicts = (probs.view(num, -1) > 0.5).float()
    y_gt = y_gt.view(num, -1)
    # intersection = torch.dot(probs.view(-1), y_gt.view(-1))
    intersection = predicts * y_gt
    # score = (2.0 * intersection + smooth) / (probs.sum() + y_gt.sum() + smooth)
    score = (2.0 * intersection.sum(1)) / (predicts.sum(1) + y_gt.sum(1))
    return score.mean()

def dice_loss(y_pred, y_gt):
    return 1 - dice_ceff(y_pred, y_gt)

def bce_dice_loss(y_pred, y_gt):
    bceL = nn.BCEWithLogitsLoss()(y_pred.view(-1), y_gt.view(-1))
    diceL = dice_loss(y_pred, y_gt)
    return bceL + diceL

def test():
    y_pred = torch.FloatTensor([[0.01, 0.03, 0.02, 0.02],
                                [0.05, 0.12, 0.09, 0.07],
                                [0.89, 0.85, 0.88, 0.91],
                                [0.99, 0.97, 0.95, 0.97]])
    y_gt = torch.FloatTensor([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]])
    # print(dice_ceff(y_pred, y_gt))

if __name__ == "__main__":
    test()
