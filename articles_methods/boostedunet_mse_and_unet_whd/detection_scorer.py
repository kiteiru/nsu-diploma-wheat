import json
import time
import copy
import torch
import click
import numpy as np
import warnings

from pathlib import Path
from skimage.io import imread

from netutils.metrics.detection import recall_score, precision_score

from precode.net_utils import ( batch_ids_generator,
                                torch_long, torch_float )
from precode.net_utils.tiling import tile_iterator
from precode.models import BoostedUnetSm as UnetSm
from precode.utils import init_determenistic, init_logging, unchain
from precode.labels.spikelet_detection import DETECTION_LABELS

warnings.filterwarnings("ignore")

config = dict()

debugging_info = { 'epoch_score_trace': list(),
                   'epoch_loss_trace': list(),
                   'epoch_times': list(),
                   'max_memory_consumption': 0.,
                   'epoch_additional_score_trace': list()
                 }

def init_precode():
    from precode.augmentations.detection import train_aug, infer_aug

    config['TRAIN_AUG'] = train_aug
    config['EVAL_AUG'] = infer_aug

def init_global_config(**kwargs):
    from precode.utils.config import init_kwargs, init_device, init_verboser, init_options

    init_kwargs(config, kwargs)
    init_device(config)
    init_verboser(config)
    init_options(config)

    config['PREFIX'] = time.strftime("%H-%M:%d-%m-%y_", time.gmtime())

        
def load_data():
    data = {}
    splitpath = Path(config['SPLITPATH'])

    with open(splitpath) as f:
        split = json.load(f)
    
    for key in ['test']:
        imgpathes = list()
        maskpathes = list()
    
        for info in split[key]:
            imgpathes.append(Path(info['img']))
            maskpathes.append(Path(info['mask']))

        data[key] = (np.array(imgpathes), np.array(maskpathes))

    return data

def create_model(n_classes):
    model = UnetSm( out_channels=n_classes,
                    encoder_name=config['BACKBONE'],
                    ntree=config['NTREE'])

    model.to(config['DEVICE'])

    return model

def augmented_load(imgpathes, maskpathes, aug):
    images = list()
    masks = list()

    for idx, (imgpath, maskpath) in enumerate(zip(imgpathes, maskpathes)):
        img = imread(imgpath)
        mask = imread(maskpath).clip(max=1)

        auged = aug(image=img, mask=mask)

        image = auged['image'].transpose(2, 0, 1)

        images.append(image)
        masks.append(auged['mask'])

    return np.array(images), np.array(masks)


def apply_image_augmentations(imgs, aug):
    auged_imgs = list()

    for img in imgs:
        auged = aug(image=img)

        aug_img = auged['image']

        auged_imgs.append(aug_img.transpose(2, 0, 1))

    return np.array(auged_imgs)


def detection_score(mask, pred_mask, distance):
    recall, = recall_score(mask, pred_mask, 'binary', distance=distance)
    precision, = precision_score(mask, pred_mask, 'binary', distance=distance)

    return precision, recall

def eval(model, imgpathes, maskpathes):   
    model.eval()

    precisions = list()
    recalls = list()
    f1s = list()

    IDX = 0

    test = None
    data_org = "random"

    with open("splits/" + data_org + ".json", 'rb') as f:
        test = json.load(f)
    test = list(test["test"])

    for imgpath, maskpath in config['VERBOSER'](zip(imgpathes, maskpathes)):
        img = imread(imgpath)
        mask = imread(maskpath).clip(max=1)

        pred_mask = np.zeros_like(mask)
        prob_mask = np.zeros_like(mask, dtype=np.float32)

        shape = mask.shape
        tile_shape = (384, 384)
        steps = tile_shape

        with torch.no_grad():
            for selections in unchain(tile_iterator(shape, tile_shape, steps), config['BATCH_SIZE']):
                imgs_batch = [ img[selector] for selector in selections ]
                imgs_batch = np.array(imgs_batch)
                imgs_batch = apply_image_augmentations(imgs_batch, config['EVAL_AUG'])
                imgs_batch = torch_float(imgs_batch, config['DEVICE'])

                logits_batch = model(imgs_batch)
                pred_mask_batch = logits_batch.argmax(axis=1)

                pred_mask_batch = pred_mask_batch.cpu().data.numpy()
                prob_mask_batch = torch.softmax(logits_batch, axis=1)[:, 1].cpu().data.numpy()

                for pred_mask_tile, prob_mask_tile, selector in zip(pred_mask_batch, prob_mask_batch, selections):
                    pred_mask[selector] = pred_mask_tile
                    prob_mask[selector] = prob_mask_tile

                ## score mask pred_mask
            from skimage.io import imsave

            # colored_prob_mask = np.zeros((*mask.shape, 3))
            # colored_prob_mask[:, :, 0] = 255 * prob_mask
            # colored_prob_mask[:, :, 2] = 255 * (1 - prob_mask)
            # colored_prob_mask = colored_prob_mask.astype(np.uint8)

            name = str(Path(test[IDX]["img"]).stem)
            imsave(f'test_whd/' + data_org + f'/{name}.png', 255 * pred_mask)
            # imsave(f'tmp/prob_mask_{IDX}.png', colored_prob_mask)
            imsave(f'test_whd/' + data_org + f'/{name}_true.png', 255 * mask)
            # imsave(f'tmp/img_{IDX}.jpg', img)

            distance = 2 * np.sqrt(config['COEFS'][imgpath.stem]) / config['RATIOS'][imgpath.stem]

            precision, recall = detection_score(mask, pred_mask, distance)
            precisions.append(precision)
            recalls.append(recall)

            if precision + recall != 0:
                f1 = 2 * (precision*recall) / (precision+recall)
            else:
                f1 = 0

            f1s.append(f1)

            IDX +=1

    score = np.mean(f1s)

    additional = {
        'f1s': np.mean(f1s),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls)
    }

    config['LOGGER'].info(f'score - {100 * score:.2f}%')

    for key in additional:
        config['LOGGER'].info(f'{key} - {100 * additional[key]:.2f}%')

    print(f"\n\n{round(np.mean(precisions),3)} {round(np.mean(recalls),3)} {round(np.mean(f1s),3)}")
            

#     return score, ({
#         'f1s': np.mean(f1s),
#         'precision': np.mean(precisions),
#         'recall': np.mean(recalls)
#     })

def load(model):
    state = torch.load(config['MODELNAME'], map_location=config['DEVICE'])

    model.load_state_dict(state)

def init_labels():
    config['LABEL_MAP'] = DETECTION_LABELS

    return len(config['LABEL_MAP'])

@click.command()
@click.option('--splitpath', '-sp', type=str)
@click.option('--backbone', '-bone', type=str, default='efficientnet-b2')
@click.option('--batch_size', '-bs', type=int, default=16)
@click.option('--ntree', type=int, default=1)
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

    with open('coefs.json', 'r') as f:
        config['COEFS'] = json.load(f)
        
    with open('ratios.json', 'r') as f:
        config['RATIOS'] = json.load(f)

    config['LOGGER'].info(f'start load data')
    data = load_data()

    config['LOGGER'].info(f'create model')
    model = create_model(n_classes=n_classes)
    
    config['LOGGER'].info(f'load model')
    load(model)

    config['LOGGER'].info(f'eval model')
    eval(model, *data['test'])

if __name__ == '__main__':
    main()
