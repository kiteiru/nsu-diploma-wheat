import torch
import numpy as np

from itertools import product

from losses_pytorch.utils import __reduction_loss

def weighted_hausdorff_distance_with_probs_loss( y_points_batch,
                                                 pred_probs_batch,
                                                 *,
                                                 reduction='mean',
                                                 eps=1e-3,
                                                 alpha=-9 ):

    assert len(pred_probs_batch.shape) == 3
    device = pred_probs_batch.device

    assert len(y_points_batch) == pred_probs_batch.shape[0]
    batch_size = len(y_points_batch)

    height, width = pred_probs_batch.shape[1:]

    maximal_distance = np.sqrt(height**2 + width**2)
    maximal_distance = torch.tensor(maximal_distance)

    coordinates = np.array(
        [ np.array(elem) for elem in
          product(np.arange(height),
                  np.arange(width)) ]
    )

    coordinates = torch.from_numpy(coordinates)
    coordinates = coordinates.to(device, dtype=torch.float)

    whd_loss_batch = list()

    for idx in np.arange(batch_size):
        points = y_points_batch[idx]
        points = torch.from_numpy(points)
        points = points.to(device, dtype=torch.float)

        pred_probs = pred_probs_batch[idx]

        # no points on image
        if len(points) == 0:
            whd_loss_batch.append(maximal_distance)
            continue

        def compute_distance_matrix(x, y):
            diff = x.unsqueeze(1) - y.unsqueeze(0)
            norm_matrix = torch.sum(diff**2, dim=-1)
            distance_matrix = torch.sqrt(norm_matrix)

            return distance_matrix

        distance_matrix = compute_distance_matrix(coordinates, points)
        min_distances, _ = torch.min(distance_matrix, dim=1)

        pred_probs_flatten = torch.flatten(pred_probs, start_dim=0)
        square = torch.sum(pred_probs_flatten)

        first_term = torch.sum(min_distances*pred_probs_flatten) / ( square + eps )

        pred_probs_flatten = pred_probs_flatten.unsqueeze(1)
        map_ = pred_probs_flatten*distance_matrix + (1-pred_probs_flatten)*maximal_distance

        second_term = torch.mean(torch.mean((map_+eps)**alpha, dim=0)**(1/alpha))

        whd_loss_batch.append(first_term+second_term)

    whd_loss_batch = torch.stack(whd_loss_batch)

    whd_loss = __reduction_loss(whd_loss_batch, reduction)

    return whd_loss
