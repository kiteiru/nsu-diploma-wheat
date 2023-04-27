import torch
import torch.nn.functional as F

from nn_tools.losses.misc import __reduction_loss, __shape_assertion


def mse_with_logits_loss( y_masks_batch,
                          pred_logits_batch,
                          *,
                          reduction='mean',
                          nclasses=None ):
    """ y_masks_batch size [batch_size, height, width]
        pred_logits_batch [batch_size, nclasses, height, width]
    """

    assert reduction in {'sum', 'mean'}

    __shape_assertion(y_masks_batch, pred_logits_batch)

    if nclasses is not None:
        assert pred_logits_batch.shape[1] == nclasses
    else:
        nclasses = pred_logits_batch.shape[1]

    assert nclasses in {1, 2}

    if nclasses == 1:
        pred_probs_batch = F.sigmoid(pred_logits_batch)
    else:
        pred_probs_batch = F.softmax(pred_logits_batch, dim=1)

    if nclasses == 1:
        pred_probs_batch = pred_probs_batch.squeeze(1)
        mse_loss_batch = F.mse_loss(pred_probs_batch, y_masks_batch, reduction='none')
    else:
        pred_probs_batch = pred_probs_batch[:, 1]
        mse_loss_batch = F.mse_loss(pred_probs_batch, y_masks_batch, reduction='none')

    mse_loss_batch = torch.sum(mse_loss_batch, dim=(1, 2))
    mse_loss = __reduction_loss(mse_loss_batch, reduction)

    return mse_loss
