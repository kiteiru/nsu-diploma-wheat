import json
import time
import copy
import torch
import click
import numpy as np
import imagesize


from pathlib import Path
from skimage.io import imread, imsave
from tqdm import tqdm

from precode.net_utils import batch_ids_generator, torch_long, torch_float
from precode.net_utils.tiling import tile_iterator
from precode.utils import init_logging, unchain
from precode.labels.spikelet_detection import DETECTION_LABELS

from nn_tools.metrics.detection import recall_score, precision_score
from nn_tools.models import BoostedUnetSm as UnetSm
from nn_tools.process.pre import apply_image_augmentations, apply_only_image_augmentations
from nn_tools.utils import init_determenistic

config = dict()

debugging_info = { 'epoch_score_trace': list(),
                   'epoch_loss_trace': list(),
                   'epoch_times': list(),
                   'max_memory_consumption': 0.,
                   'epoch_additional_score_trace': list()
                 }

def init_precode():
    from precode.augmentations.detection import infer_aug

    config['EVAL_AUG'] = infer_aug

def init_global_config(**kwargs):
    from precode.utils.config import init_kwargs, init_device, init_verboser, init_options

    init_kwargs(config, kwargs)
    init_device(config)
    init_verboser(config)
    init_options(config)

    config['PREFIX'] = time.strftime("%H-%M:%d-%m-%y_", time.gmtime())

def load_data():
    imgpathes = list()

    if config['IMGPATH']:
        imgpath = Path(config['IMGPATH'])

        for i in (Path(imgpath)).glob('*.[j|J][p|P][g|G]'):
            imgpathes.append(i)

    shapes = list()

    for imgpath in imgpathes:
        shape = imagesize.get(str(imgpath))
        shapes.append(shape[::-1])

    data = {}

    data['infer'] = {
        'shapes': shapes,
        'imgs': imgpathes,
    }

    return data

def create_model(n_classes):
    model = UnetSm( out_channels=n_classes,
                    encoder_name=config['BACKBONE'],
                    ntree=config['NTREE'])

    model.to(config['DEVICE'])

    return model

def eval(model, data):   
    model.eval()

    imgpathes = data['infer']['imgs']

    for imgpath in tqdm(config['VERBOSER'](imgpathes)):
        img = imread(imgpath)

        pred_mask = np.zeros(img.shape[:-1])

        shape = pred_mask.shape
        tile_shape = (384, 384)
        steps = tile_shape

        with torch.no_grad():
            for selections in unchain(tile_iterator(shape, tile_shape, steps), config['BATCH_SIZE']):
                imgs_batch = [ img[selector] for selector in selections ]
                imgs_batch = np.array(imgs_batch)
                imgs_batch = apply_only_image_augmentations(imgs_batch, config['EVAL_AUG'])
                imgs_batch = torch_float(imgs_batch, config['DEVICE'])

                logits_batch = model(imgs_batch)
                pred_mask_batch = logits_batch.argmax(axis=1)

                pred_mask_batch = pred_mask_batch.cpu().data.numpy()

                for pred_mask_tile, selector in zip(pred_mask_batch, selections):
                    pred_mask[selector] = pred_mask_tile

            outdir = Path.cwd() / config['OUTPATH']

            if not outdir.is_dir():
                outdir.mkdir()

            imsave( outdir / f'{imgpath.stem}.png', 255 * pred_mask)

def load(model):
    state = torch.load(config['MODELNAME'], map_location=config['DEVICE'])

    model.load_state_dict(state)

def init_labels():
    config['LABEL_MAP'] = DETECTION_LABELS

    return len(config['LABEL_MAP'])

@click.command()
@click.option('--outpath', '-op', type=str, default='.')
@click.option('--imgpath', '-ip', type=str)
@click.option('--backbone', '-bone', type=str, default='efficientnet-b2')
@click.option('--batch_size', '-bs', type=int, default=16)
@click.option('--ntree', type=int, default=3)
@click.option('--verbose', is_flag=True)
@click.option('--modelname', '-mn', type=str)
def main(**kwargs):
    init_determenistic()
    init_precode()

    init_global_config(**kwargs)
    init_logging(config, __name__)

    n_classes = init_labels()

    config['LOGGER'].info(f'n_classes {n_classes}')

    for key in config:
        if key != 'LOGGER':
            config['LOGGER'].info(f'{key} {config[key]}')
            debugging_info[key.lower()] = str(config[key])

    config['LOGGER'].info(f'start load data')
    data = load_data()

    config['LOGGER'].info(f'create model')
    model = create_model(n_classes=n_classes)

    config['LOGGER'].info(f'load model')
    load(model)

    config['LOGGER'].info(f'eval model')
    eval(model, data)

if __name__ == '__main__':
    main()
