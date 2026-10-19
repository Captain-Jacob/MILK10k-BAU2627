import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss to handle multi-label class imbalance."""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_pos > 0:
            los_pos *= (1 - xs_pos) ** self.gamma_pos
        if self.gamma_neg > 0:
            los_neg *= xs_pos ** self.gamma_neg

        loss = los_pos + los_neg
        return -loss.sum()