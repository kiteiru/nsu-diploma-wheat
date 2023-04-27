import yaml
import torch
import random
import imgaug
import numpy as np

import logging as log

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

def init_logging(config, name):
    config['LOGGER'] = log.getLogger(name)
    config['LOGGER'].setLevel(log.INFO)

    config['LOGGER'].handlers.clear()

    handler = log.StreamHandler()

    formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    config['LOGGER'].addHandler(handler)

def load_yaml_config(path):
    assert path.is_file()

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return data
