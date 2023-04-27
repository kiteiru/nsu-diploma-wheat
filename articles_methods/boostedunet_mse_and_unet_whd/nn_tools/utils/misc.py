import os
import yaml
import torch
import random
import imgaug
import numpy as np

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

    os.environ['PYTHONHASHSEED'] = str(seed)

def load_yaml_config(path):
    assert path.is_file()

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return data