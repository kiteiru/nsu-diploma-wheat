import os
os.environ['PYTORCH_JIT'] = '0'

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
from precode.utils import unchain
from precode.labels.spikelet_detection import DETECTION_LABELS

import nn_tools.utils.config as cfg
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
    cfg.init_timestamp(config)
    cfg.init_run_command(config)
    cfg.init_kwargs(config, kwargs)
    cfg.init_logging(config, __name__, config['LOGGER_TYPE'], filename=config['PREFIX']+'logger_name.txt')
    cfg.init_device(config)
    cfg.init_verboser(config, logger=config['LOGGER'])
    cfg.init_options(config)

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

from skimage.transform import resize


def crop_data(img, mask):

    xc, yc = np.where(mask > 0)

    xs = np.min(xc) - int(0.05 * img.shape[0])
    xe = np.max(xc) + int(0.05 * img.shape[0])

    dx = xe - xs

    if xs < 0:
        xs = 0
        xe = dx

    if xe > img.shape[0]:
        xe = img.shape[0]
        xs = xe - dx

    ys = np.min(yc) + 0.5 * (np.max(yc) - np.min(yc)) - dx // 2
    ys = int(ys)

    if ys + xe -xs > img.shape[1]:
        ye = img.shape[1]
    else:
        ye = ys + xe -xs

    ys = ye - (xe -xs)

    if ys < 0:
        ye -= ys
        ys = 0

    if ye - ys > img.shape[1]:
        ys = 0
        ye = img.shape[1]

    if xe - xs > img.shape[0]:
        xs = 0
        xe = img.shape[0]

    if ye - ys != xe -xs:
        if ye - ys < xe -xs:
            m = (xe - xs - (ye - ys)) // 2
            xe -= m
            xs = xe - (ye - ys)
        else:
            m = (ye - ys - (xe - xs)) // 2
            ye -= m
            ys = ye - (xe - xs)

    cropped = img[xs:xe, ys:ye]

    cropped = resize(cropped, (384, 384)) * 255
    cropped = cropped.astype(np.uint8)

    size = xe - xs

    return cropped, size, (slice(xs, xe), slice(ys, ye))


def eval(model, data):
    model.eval()

    imgpathes = data['infer']['imgs']

    for imgpath in tqdm(config['VERBOSER'](imgpathes)):
        img = imread(imgpath)
        mask = imread(imgpath.with_suffix('.png'))

        cropped_img, orig_size, slices = crop_data(img, mask)

        pred_mask = np.zeros(cropped_img.shape[:-1])

        shape = pred_mask.shape
        tile_shape = (384, 384)
        steps = tile_shape

        with torch.no_grad():
            for selections in unchain(tile_iterator(shape, tile_shape, steps), config['BATCH_SIZE']):
                imgs_batch = [ cropped_img[selector] for selector in selections ]
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

            # imsave( outdir / f'{imgpath.stem}_grain_crop.png', cropped_img)
            # imsave( outdir / f'{imgpath.stem}_grain_mask.png', 255 * pred_mask)


            mask = np.zeros(img.shape[:-1])
            pred_mask = resize(pred_mask, (orig_size, orig_size))
            pred_mask[pred_mask > 0] = 1

            mask[slices] = pred_mask



            imsave( outdir / f'{imgpath.stem}_grain.png', 255 * mask)

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
@click.option('--logger_type', '-lt', type=click.Choice(['stream', 'file'], case_sensitive=False), default='stream')
@click.option('--verbose', is_flag=True)
@click.option('--modelname', '-mn', type=str)
def main(**kwargs):
    init_determenistic()
    init_precode()

    init_global_config(**kwargs)

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