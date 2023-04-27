import torch
import random
import imgaug
import numpy as np

def batch_ids_generator(size, batch_size, shuffle=False):
    ids = np.arange(size)

    if shuffle:
        np.random.shuffle(ids)

    poses = np.arange(batch_size, size, batch_size)
    return np.split(ids, poses)

def init_determenistic(seed=1996, precision=10):
    """ NOTE options
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determenistic = True
        may lead to numerical unstability
    """
    random.seed(seed)
    np.random.seed(seed)
    imgaug.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determenistic = True
    torch.backends.cudnn.enabled = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=precision)
