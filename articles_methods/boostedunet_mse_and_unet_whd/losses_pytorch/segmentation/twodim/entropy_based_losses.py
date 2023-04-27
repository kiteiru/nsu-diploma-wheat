import torch
import torch.nn.functional as F

from losses_pytorch.utils import __reduction_loss

def cross_entropy_with_logits_loss( y_masks_batch,
                                    pred_logits_batch,
                                    *,
                                    loss_masks_batch=None,
                                    reduction='mean',
                                    nclasses=None,
                                    activation='softmax',
                                    weight=None ):
    """ y_masks_batch size [batch_size, height, width]
        pred_logits_batch [batch_size, nclasses, height, width]
        loss_masks_batch size [batch_size, height, width]
    """

    assert reduction in {'sum', 'mean'}

    assert len(y_masks_batch.shape) == 3
    assert len(pred_logits_batch.shape) == 4

    assert y_masks_batch.shape[0] == pred_logits_batch.shape[0]
    assert y_masks_batch.shape[1] == pred_logits_batch.shape[2]
    assert y_masks_batch.shape[2] == pred_logits_batch.shape[3]

    if loss_masks_batch is not None:
        assert y_masks_batch.shape == loss_masks_batch.shape
        loss_mask_unique = torch.unique(loss_masks_batch, sorted=True)

        if loss_mask_unique.size()[0] == 1:
            assert loss_mask_unique[0] == 0 or loss_mask_unique[0] == 1
        elif loss_mask_unique.size()[0] == 2:
            assert loss_mask_unique[0] == 0 and loss_mask_unique[1] == 1
        else:
            assert 0

    if nclasses is not None:
        assert pred_logits_batch.shape[1] == nclasses
    else:
        nclasses = pred_logits_batch.shape[1]

    if nclasses == 1:
        pred_logits_batch = pred_logits_batch.squeeze(1)
        ce_loss_batch = F.binary_cross_entropy_with_logits(pred_logits_batch, y_masks_batch, reduction='none', weight=weight)
    else:
        if activation == 'softmax':
            ce_loss_batch = F.cross_entropy(pred_logits_batch, y_masks_batch, reduction='none', weight=weight)
        elif activation == 'sigmoid':
            raise NotImplemented()
        else:
            raise ValueError()

    if loss_masks_batch is not None:
        ce_loss_batch = ce_loss_batch * loss_masks_batch

    ce_loss_batch = torch.sum(ce_loss_batch, dim=(1, 2))
    ce_loss = __reduction_loss(ce_loss_batch, reduction)

    return ce_loss
