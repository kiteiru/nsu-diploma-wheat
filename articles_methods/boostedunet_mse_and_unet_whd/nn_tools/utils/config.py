import sys
import time
import torch
import logging as log

from tqdm import tqdm
from pathlib import Path

from nn_tools.utils import load_yaml_config
from nn_tools.utils.tqdm import TqdmLogger

def init_logging(config, name, logtype='stream', **kwargs):
    config['LOGGER'] = log.getLogger(name)
    config['LOGGER'].setLevel(log.INFO)

    config['LOGGER'].handlers.clear()

    if logtype == 'stream':
        handler = log.StreamHandler()
    elif logtype == 'file':
        handler = log.FileHandler( kwargs.get('filename'),
                                   mode='a',
                                   encoding='utf-8' )
    else:
        handler = log.NullHandler()

    formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    config['LOGGER'].addHandler(handler)

def init_kwargs(config, kwargs):
    for key, value in kwargs.items():
        if key.upper().endswith('PATH'):
            if value.upper() == 'SKIP':
                continue
            elif value is not None:
                config[key.upper()] = Path(value)
            else:
                raise RuntimeError(f'Path option {key} is None')
        else:
            config[key.upper()] = value

def init_device(config):
    if torch.cuda.is_available():
        config['DEVICE'] = torch.device('cuda')
    else:
        config['DEVICE'] = torch.device('cpu')

def init_verboser(config, **kwargs):
    if config['VERBOSE']:
        file_ = TqdmLogger(kwargs.get('logger')) if 'logger' in kwargs else None
        config['VERBOSER'] = lambda x: tqdm(x, file=file_)
    else:
        config['VERBOSER'] = lambda x: x

def init_options(config):
    for key in list(config.keys()):
        if key.endswith('OPTIONS_PATH'):
            option_key = key[:-5]
            config[option_key] = load_yaml_config(config[key])

def init_run_command(config):
    config['SCRIPT'] = ' '.join(sys.argv)

def init_timestamp(config):
    config['PREFIX'] = time.strftime("%d-%m-%y:%H-%M_", time.gmtime())
