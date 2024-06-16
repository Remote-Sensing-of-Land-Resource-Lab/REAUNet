import torch
from torch import nn


# loss for Edge Detection
class WBCELoss(nn.Module):
    def __init__(self, balance=1.1):
        super().__init__()
        self.balance = balance

    def forward(self, output, target):
        device = output.device
        n, c, h, w = output.size()
        weights = torch.zeros(size=(n, c, h, w), device=device)
        for i in range(n):
            t = target[i, :, :, :]
            pos = (t == 1).sum()
            neg = (t == 0).sum()
            valid = neg + pos
            weights[i, t == 1] = neg / valid
            weights[i, t == 0] = pos * self.balance / valid
        loss_bce = nn.BCELoss(weights, reduction='sum')(output, target)
        return loss_bce / n


# loss for Segmentation
class BCEDiceLoss(nn.Module):
    def __init__(self, frac=0.5, threshold=0.5, eps=1e-10):
        super().__init__()
        self.frac = frac
        self.threshold = threshold
        self.eps = eps

    def forward(self, output, target):
        # bce loss
        loss_bce = nn.BCELoss()(output, target)
        batch = output.shape[0]
        output = output.reshape(batch, -1)
        target = target.reshape(batch, -1)
        # dice loss
        output_t = torch.where(output > self.threshold, torch.ones_like(output), torch.zeros_like(output))
        inter = (output_t * target).sum(-1)
        union = (output_t + target).sum(-1)
        dice = (2 * inter + self.eps) / (union + self.eps)
        loss_dice = 1 - (dice.sum() / batch)
        return self.frac * loss_bce + loss_dice
