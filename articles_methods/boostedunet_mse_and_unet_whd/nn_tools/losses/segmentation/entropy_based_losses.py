import torch
import torch.nn.functional as F

from nn_tools.losses.misc import __reduction_loss, __shape_assertion

def cross_entropy_with_logits_loss( y_masks_batch,
                                    pred_logits_batch,
                                    *,
                                    reduction='mean',
                                    nclasses=None,
                                    class_weights=None,
                                    activation='softmax' ):
    """ y_masks_batch size [batch_size, height, width]
        pred_logits_batch [batch_size, nclasses, height, width]
    """

    assert reduction in {'sum', 'mean'}

    __shape_assertion(y_masks_batch, pred_logits_batch)

    if nclasses is not None:
        assert pred_logits_batch.shape[1] == nclasses
    else:
        nclasses = pred_logits_batch.shape[1]

    if class_weights is not None:
        assert class_weights.shape[0] == nclasses

    if nclasses == 1:
        pred_logits_batch = pred_logits_batch.squeeze(1)
        ce_loss_batch = F.binary_cross_entropy_with_logits(pred_logits_batch, y_masks_batch, weight=class_weights, reduction='none')
    else:
        if activation == 'softmax':
            ce_loss_batch = F.cross_entropy(pred_logits_batch, y_masks_batch, weight=class_weights, reduction='none')
        elif activation == 'sigmoid':
            raise NotImplemented()
        else:
            raise ValueError()

    ce_loss_batch = torch.sum(ce_loss_batch, dim=(1, 2))
    ce_loss = __reduction_loss(ce_loss_batch, reduction)

    return ce_loss

def focal_with_logits_loss( y_masks_batch,
                            pred_logits_batch,
                            *,
                            gamma=1,
                            reduction='mean',
                            nclasses=None,
                            class_weights=None,
                            activation='softmax' ):
    """ y_masks_batch size [batch_size, height, width]
        pred_logits_batch [batch_size, nclasses, height, width]

        NOTE:
            based on arxiv.org/pdf/1708.02002.pdf
    """

    assert reduction in {'sum', 'mean'}

    __shape_assertion(y_masks_batch, pred_logits_batch)

    if nclasses is not None:
        assert pred_logits_batch.shape[1] == nclasses
    else:
        nclasses = pred_logits_batch.shape[1]

    if class_weights is not None:
        assert class_weights.shape[0] == nclasses

    if nclasses == 1:
        raise NotImplemented()
    else:
        if activation == 'softmax':
            ce_loss_batch = F.cross_entropy(pred_logits_batch, y_masks_batch, weight=class_weights, reduction='none')

            if class_weights is not None:
                ce_loss_batch_ = F.cross_entropy(pred_logits_batch, y_masks_batch, reduction='none')
                pred_probs_batch = torch.exp(-ce_loss_batch_)
            else:
                pred_probs_batch = torch.exp(-ce_loss_batch)

            fl_loss_batch = ce_loss_batch * (1 - pred_probs_batch)**gamma
        elif activation == 'sigmoid':
            raise NotImplemented()
        else:
            raise ValueError()

    fl_loss_batch = torch.sum(fl_loss_batch, dim=(1, 2))
    fl_loss = __reduction_loss(fl_loss_batch, reduction)

    return fl_loss