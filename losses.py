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
    

class Survival_Loss(nn.Module):
    def __init__(self):
        super(Survival_Loss, self).__init__()

    def forward(self, t_pred, t, c):
        """"
        Survival loss function for continuous survival analysis.
        t_pred: predicted survival time
        t: true survival time
        c: censoring indicator (1 if event occurred, 0 if censored)
        """
        loss = torch.abs(t_pred - t) * c + (1 - c) * F.relu(t_pred - t)
        return loss.mean()
        


def survival_log_likelihood_loss(event_probs, cumulative_probs, time_bins, event_indicators):
    """
    event_probs: tensor of shape [batch_size, T]
    cumulative_probs: tensor of shape [batch_size, T]
    time_bins: tensor of ints of shape [batch_size] (which time bin the event/censoring happened)
    event_indicators: tensor of bools of shape [batch_size] (True if event occurred)
    """
    batch_size = event_probs.size(0)
    losses = []
    for i in range(batch_size):
        t = time_bins[i]
        if event_indicators[i]:
            # Event occurred at t
            prob = event_probs[i, t]
        else:
            # Censored at t → we want P(survival up to t)
            prob = 1.0 - cumulative_probs[i, t]
        losses.append(-torch.log(prob + 1e-8))  # add epsilon for numerical stability

    return torch.stack(losses).mean()


def log_likelihood_loss(event_probs, event_time_bins, event_indicators):
    """
    event_probs: Tensor of shape [batch_size, n_follow_up] with non-cumulative probabilities
    event_time_bins: LongTensor of shape [batch_size], values in the range [0, n_follow_up-1]
    event_indicators: BoolTensor of shape [batch_size], 1 if treatment, 0 if active surveillance
    """
    batch_size = event_probs.size(0)
    losses = []

    for i in range(batch_size):
        t = event_time_bins[i].item()
        if event_indicators[i] == 1:
            prob = event_probs[i, t]
        else:
            prob = event_probs[i, t:].sum()
        losses.append(-torch.log(prob + 1e-8))

    return torch.stack(losses).mean()


def log_likelihood_loss_vectorized(event_probs, event_time_bins, event_indicators):
    """
    Vectorized version of above.
    event_probs: [batch_size, 5]
    event_time_bins: [batch_size] (LongTensor)
    event_indicators: [batch_size] (BoolTensor or FloatTensor)
    """
    batch_size, num_bins = event_probs.shape

    arange_bins = torch.arange(num_bins).unsqueeze(0).to(event_time_bins.device)  # [1, 5]
    # Create a mask for survival beyond observed time (for censored)
    mask = arange_bins >= event_time_bins.unsqueeze(1)  # [batch_size, 5]

    # Prob for observed event
    p_event = event_probs[torch.arange(batch_size), event_time_bins]

    # Prob of survival (sum of probs from time bin onward)
    p_survival = (event_probs * mask).sum(dim=1)

    # Combine both using indicator
    loss = -(
        event_indicators * torch.log(p_event + 1e-8) +
        (1 - event_indicators.float()) * torch.log(p_survival + 1e-8)
    )

    return loss.mean()


class CEAttentionEntropyLoss(nn.Module):
    """
    Cross-entropy loss with attention entropy regularization.
    """
    def __init__(self, alpha=0.1):
        super(CEAttentionEntropyLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, attention_weights):
        ce_loss = self.ce_loss(logits, labels)
        
        attention_entropy = torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1).mean()
        return ce_loss + self.alpha * attention_entropy


class CEKLDivLoss(nn.Module):
    """
    Cross-entropy loss with KL divergence regularization between attention and probabilites.
    """
    def __init__(self, alpha=0.1, eps=1e-8):
        super(CEKLDivLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, a, p=None):
        ce_loss = self.ce_loss(logits, labels)

        if p is None:
            return ce_loss
        else:
            p_norm = p / (p.sum() + self.eps)
        
            kl_div = (p_norm * (p_norm + self.eps).log() - p_norm * (a + self.eps).log()).sum()

            return ce_loss + self.alpha * kl_div

