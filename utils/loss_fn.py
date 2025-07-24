import torch
import torch.nn as nn

def get_loss_fn(name: str, **kwargs):
    name = name.lower()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)

    elif name == "bce":
        return nn.BCELoss(**kwargs)

    elif name == "bce_logits":
        return nn.BCEWithLogitsLoss(**kwargs)

    elif name == "mse":
        return nn.MSELoss(**kwargs)

    elif name == "mae":
        return nn.L1Loss(**kwargs)

    elif name == "smooth_l1":
        return nn.SmoothL1Loss(**kwargs)

    elif name == "focal":
        return FocalLoss(**kwargs)

    else:
        raise ValueError(f"Unsupported loss function: {name}")
    
# 클래스 불균형이 심할 때, 쉬운 예제는 무시하고 어려운 예제에 집중하도록 만든 손실 함수
# alpha -> 양성 클래스의 비중 / gamma -> 쉬운 예제 무시 정도 / reduction -> mean or sum
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # inputs: logits, targets: binary labels
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss