import torch
from torch import nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)

def mask_loss(pred, target, mask):
    rmse = RMSELoss()
    mask = mask.bool()
    mask_pred = pred[mask]
    mask_target = target[mask]
    loss = rmse(mask_pred, mask_target)
    loss = torch.nan_to_num(loss, nan=0.0)
    return loss

def criterion(pred, ctv, bladder, rectum, applicator, target):
    ctv_loss = mask_loss(pred, target, ctv)
    bladder_loss = mask_loss(pred, target, bladder)
    rectum_loss = mask_loss(pred, target, rectum)
    applicator_loss = mask_loss(pred, target, applicator)
    
    l1_loss = RMSELoss()(pred, target)
    return ctv_loss + 0.4 * bladder_loss + 0.4 * rectum_loss + 0.6 * applicator_loss + 0.2 * l1_loss
