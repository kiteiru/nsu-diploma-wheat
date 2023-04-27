import torch.nn.functional as F

def cross_entropy_with_logits_loss(y_labels_batch, pred_logits_batch, *, reduction='mean', nclasses=None):
    """ y_labels_batch size [batch_size, ]
        pred_logits_batch [batch_size, nclasses]
    """

    assert reduction in {'sum', 'mean'}

    assert y_labels_batch.shape[0] == pred_logits_batch.shape[0]

    if nclasses is not None:
        assert pred_logits_batch.shape[1] == nclasses

    ce_loss = F.cross_entropy(pred_logits_batch, y_labels_batch, reduction=reduction)

    return ce_loss
