import torch
import torch.nn.functional as F

def kld_loss(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Calculates the kld loss
    """
    b, c, h, w = outputs.shape

    outputs = outputs.view(-1)
    outputs_ = 1 -outputs
    outputs = torch.stack([outputs, outputs_], dim=1)
    outputs = outputs.log()

    labels=F.interpolate(labels, size=(h, w), mode="bilinear", align_corners=False)

    labels = labels.view(-1)
    labels_ = 1 - labels
    labels = torch.stack([labels, labels_], dim=1)

    return F.kl_div(outputs, labels, reduction='batchmean')

def bce(outputs: torch.Tensor, labels: torch.Tensor):
    b, c, h, w = outputs.shape

    labels=F.interpolate(labels, size=(h, w), mode="bilinear", align_corners=False)

    loss = torch.nn.BCELoss()(outputs, labels)
    # loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
    return loss
