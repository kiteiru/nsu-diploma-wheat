import torch
import torch.nn.functional as F

from torch.autograd import Variable

from nn_tools.losses.misc import __reduction_loss, __shape_assertion

 

def discriminative_loss( y_masks_batch,
                         pred_logits_batch,
                         *,
                         reduction='mean',
                         delta_var=0.5,
                         delta_dist=1.5,
                         norm=2,
                         alpha=1.0,
                         beta=1.0,
                         gamma=0.001 ):
    """ y_masks_batch size [batch_size, height, width]
        pred_logits_batch [batch_size, n_features, height, width]

        NOTE
            described in https://arxiv.org/pdf/1708.02551.pdf

            n_max_objects == n_features

            delta_var - max distance for pixel attractive forces inside object
            delta_dist - max distance for repulsing forces between objects
    """

    assert reduction in {'sum', 'mean'}
    assert norm in {1, 2}

    __shape_assertion(y_masks_batch, pred_logits_batch)

    batch_size, n_features, height, width = pred_logits_batch.shape
    assert torch.max(y_masks_batch) < n_features

    y_masks_batch = y_masks_batch.view(batch_size, height*width)
    pred_logits_batch = pred_logits_batch.view(batch_size, n_features, height*width)

    n_objects, _ = torch.max(y_masks_batch, dim=-1)

    object_centers = get_object_centers(y_masks_batch, pred_logits_batch, n_objects)

    variance_term = __variance_term( y_masks_batch,
                                     pred_logits_batch,
                                     object_centers,
                                     n_objects,
                                     norm,
                                     delta_var )

    distance_term =  __distance_term(object_centers, n_objects, norm, delta_dist)
    regularization_term = __regularization_term(object_centers, n_objects, norm)

    disc_loss_batch = alpha * variance_term + beta * distance_term + gamma * regularization_term
    disc_loss = __reduction_loss(disc_loss_batch, reduction)

    return disc_loss

def __object_centers(y_masks_batch, pred_logits_batch, n_objects):
    """ y_masks_batch size [batch_size, height * width]
        pred_logits_batch [batch_size, n_features, height * width]
        n_objects [batch_size, ]
    """

    batch_size, n_features, height_width = pred_logits_batch.shape
    n_max_objects = n_features

    onehoted_y_masks_batch = F.one_hot(y_masks_batch, num_classes=n_max_objects+1)
    onehoted_y_masks_batch = onehoted_y_masks_batch[:, :, 1:]

    n_object_points = torch.sum(onehoted_y_masks_batch, dim=1)
    n_object_points[n_object_points==0] = 1

    all_vecs_by_objects = pred_logits_batch.unsqueeze(1).expand(batch_size, n_max_objects, n_features, height_width)
    vecs_by_objects = (all_vecs_by_objects * onehoted_y_masks_batch.permute(0, 2, 1).unsqueeze(2))

    object_centers = torch.sum(vecs_by_objects, dim=3) / n_object_points.unsqueeze(2)

    return object_centers

def __variance_term(y_masks_batch, pred_logits_batch, object_centers, n_objects, norm, delta_var):
    """ y_masks_batch size [batch_size, height * width]
        pred_logits_batch [batch_size, n_features, height * width]
        object_centers [batch_size, n_max_objects, n_features]
        n_objects [batch_size, ]

        NOTE
            onehoted_y_masks_batch[:, :, 1:] used for drop zeros class coding that not object
    """

    batch_size, n_features, height_width = pred_logits_batch.shape
    n_max_objects = n_features

    onehoted_y_masks_batch = F.one_hot(y_masks_batch, num_classes=n_max_objects+1)
    onehoted_y_masks_batch = onehoted_y_masks_batch[:, :, 1:]

    n_object_points = torch.sum(onehoted_y_masks_batch, dim=1) # [batch_size, n_max_objects]
    n_object_points[n_object_points==0] = 1

    diff = object_centers.permute(0, 2, 1).unsqueeze(3) - pred_logits_batch.unsqueeze(2)
    mask_diff = diff * onehoted_y_masks_batch.permute(0, 2, 1).unsqueeze(1)

    normes = torch.norm(mask_diff, dim=1, p=2)
    dist = torch.relu(normes - delta_var)**2

    summed_by_pixel_dist = torch.sum(dist, dim=2) / n_object_points
    term = torch.sum(summed_by_pixel_dist, axis=1) / n_objects

    return term

def __distance_term(object_centers, n_objects, norm, delta_dist):
    """ object_centers [batch_size, n_max_objects, n_features]
        n_objects [batch_size, ]
    """
    device = object_centers.device
    _, n_max_objects, _ = object_centers.size()

    pobject_centers = object_centers.permute(1, 2, 0)
    diff_matrix = pobject_centers - pobject_centers.unsqueeze(1)

    norms = torch.norm(diff_matrix , p=2, dim=-2)
    margin = 2 * delta_dist *(1 - torch.eye(n_max_objects, device=device))

    dist = torch.relu(margin.unsqueeze(-1) - norms)**2

    center_masks = torch.norm(pobject_centers, p=2, dim=-2) > 0
    dist_masks = center_masks * center_masks.unsqueeze(1)

    summed_dist = torch.sum(dist * dist_masks, dim=(0, 1))

    coefs = n_objects * (n_objects - 1)
    coefs[n_objects == 1] = 1

    term = summed_dist / coefs

    return term

def __regularization_term(object_centers, n_objects, norm):
    """ object_centers [batch_size, n_max_objects, n_features]
        n_objects [batch_size, ]
    """

    normes = torch.norm(object_centers, p=norm, dim=1)
    term = torch.sum(normes, axis=1) / n_objects

    return term
