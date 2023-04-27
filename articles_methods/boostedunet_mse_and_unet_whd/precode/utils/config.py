import torch

from tqdm import tqdm
from pathlib import Path

from precode.utils import load_yaml_config

def init_kwargs(config, kwargs):
    for key, value in kwargs.items():
        if key.upper().endswith('PATH') and value is not None:
            config[key.upper()] = Path(value)
        else:
            config[key.upper()] = value

def init_device(config):
    if torch.cuda.is_available():
        config['DEVICE'] = torch.device('cuda')
    else:
        config['DEVICE'] = torch.device('cpu')

def init_verboser(config):
    if config['VERBOSE']:
        config['VERBOSER'] = tqdm
    else:
        config['VERBOSER'] = lambda x: x

def init_options(config):
    for key in list(config.keys()):
        if key.endswith('OPTIONS_PATH'):
            option_key = key[:-5]
            config[option_key] = load_yaml_config(config[key])
