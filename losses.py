import torch
import torch.nn as nn
import torch.nn.functional as F


class Longitudinal_Loss(nn.Module):
    """
    A Loss function for longitudinal labels (Needs updates).
    Designed to ignore nans arising from missing labels
    """
    def __init__(self, reduction='sum', pos_weight=None):
        super(Longitudinal_Loss, self).__init__()
        self.reduction = reduction
        if pos_weight is None:
            self.p = 1
        else:
            self.p = pos_weight

    def forward(self, logits, labels):
        # Find non-nan indices
        idx = torch.isnan(labels) == False
        # Select only non-nan labels and their corresponding logits
        logits = logits[idx]
        labels = labels[idx]
        # Binary cross entropy loss
        L = self.p*labels*(-torch.log(F.sigmoid(logits))) + (1-labels)*(-torch.log(1-F.sigmoid(logits)))
        if torch.any(torch.isinf(L)):
            print("Inf encountered in loss")

        if self.reduction == 'sum':
            loss = torch.nansum(L)
        elif self.reduction == 'mean':
            loss = torch.nanmean(L)
        else:
            loss = L

        return loss



