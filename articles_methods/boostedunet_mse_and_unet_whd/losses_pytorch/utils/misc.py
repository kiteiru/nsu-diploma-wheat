import torch

def __reduction_metric(metric_batch, reduction):
    if reduction == 'mean':
        loss = torch.mean(1 - metric_batch, axis=0)
    elif reduction == 'sum':
        loss = torch.sum(1 - metric_batch, axis=0)
    else:
        raise RuntimeError()

    return loss

def __reduction_loss(loss_batch, reduction):
    if reduction == 'mean':
        loss = torch.mean(loss_batch, axis=0)
    elif reduction == 'sum':
        loss = torch.sum(loss_batch, axis=0)
    else:
        raise RuntimeError()

    return loss
