import torch
import torch.nn.functional as F

from nn_tools.losses.misc import __reduction_metric, __shape_assertion

def dice_with_logits_loss( y_masks_batch,
                           pred_logits_batch,
                           *,
                           smooth=1,
                           average='binary',
                           averaged_classes=None,
                           reduction='mean',
                           nclasses=None,
                           class_weights=None,
                           activation='softmax' ):
    """ y_masks_batch size [batch_size, height, width]
        pred_logits_batch [batch_size, nclasses, height, width]

        NOTE
            binary is common used implementation
            macro is described in https://ieeexplore.ieee.org/document/9433991
    """

    assert reduction in {'sum', 'mean'}

    __shape_assertion(y_masks_batch, pred_logits_batch)

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
        assert nclasses == 1 or nclasses == 2

        if nclasses == 1:
            pred_probs_batch = pred_probs_batch.squeeze()
        elif nclasses == 2:
            pred_probs_batch = pred_probs_batch[:, 1]

        dice_overlap = 2 * torch.sum(y_masks_batch * pred_probs_batch, dim=(1, 2)) + smooth
        dice_total = torch.sum(y_masks_batch + pred_probs_batch, dim=(1, 2)) + smooth

        dice_metric_batch = dice_overlap / dice_total
    elif average == 'micro':
        assert nclasses > 1

        raise NotImplemented()
    elif average == 'macro':
        assert nclasses > 1

        if averaged_classes is None:
            averaged_classes = range(1, nclasses)

        dice_metric_batch_stack = tuple()

        for class_ in averaged_classes:
            class_y_masks_batch = (y_masks_batch == class_)
            class_pred_probs_batch = pred_probs_batch[:, class_]

            dice_overlap = 2 * torch.sum(class_y_masks_batch * class_pred_probs_batch, dim=(1, 2)) + smooth
            dice_total = torch.sum(class_y_masks_batch + class_pred_probs_batch, dim=(1, 2)) + smooth

            dice_metric_batch_stack = (*dice_metric_batch_stack, dice_overlap / dice_total)

        dice_metric_batch_stack = torch.stack(dice_metric_batch_stack, dim=-1)

        if class_weights is not None:
            assert class_weights.shape[0] == len(averaged_classes)
            dice_metric_batch_stack = dice_metric_batch_stack * class_weights

        dice_metric_batch = torch.mean(dice_metric_batch_stack, dim=-1)
    else:
        raise ValueError()

    dice_loss = __reduction_metric(dice_metric_batch, reduction)

    return dice_loss
