import torch
import torch.nn.functional as F

from losses_pytorch.utils import __reduction_metric

def dice_with_logits_loss( y_masks_batch,
                           pred_logits_batch,
                           *,
                           smooth=1,
                           reduction='mean',
                           average='binary',
                           nclasses=None, 
                           activation='softmax' ):
    """ y_masks_batch size [batch_size, height, width]
        pred_logits_batch [batch_size, nclasses, height, width]
    """

    assert reduction in {'sum', 'mean'}

    assert len(y_masks_batch.shape) == 3
    assert len(pred_logits_batch.shape) == 4
    assert y_masks_batch.shape[0] == pred_logits_batch.shape[0]
    assert y_masks_batch.shape[1] == pred_logits_batch.shape[2]
    assert y_masks_batch.shape[2] == pred_logits_batch.shape[3]

    if nclasses is not None:
        assert pred_logits_batch.shape[1] == nclasses
    else:
        nclasses = pred_logits_batch.shape[1]

    if nclasses == 1:
        pred_probs_batch = F.sigmoid(pred_logits_batch)
    else:
        if activation == 'softmax':
            pred_probs_batch = F.softmax(pred_logits_batch, dim=1)
        elif activation == 'sigmoid':
            pred_probs_batch = F.sigmoid(pred_logits_batch)
        else:
            raise ValueError()

    if average == 'binary':
        assert nclasses == 1
        pred_probs_batch = pred_probs_batch.squeeze()

        dice_overlap = 2 * torch.sum(y_masks_batch * pred_probs_batch, dim=(1, 2)) + smooth
        dice_total = torch.sum(y_masks_batch + pred_probs_batch, dim=(1, 2)) + smooth

        dice = dice_overlap / dice_total
        dice_loss = __reduction_metric(dice, reduction)
    elif average == 'micro':
        assert nclasses > 1

        raise NotImplemented()
    elif average == 'macro':
        assert nclasses > 1

        raise NotImplemented()
    else:
        raise ValueError()

    return dice_loss
