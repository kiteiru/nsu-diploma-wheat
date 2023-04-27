# import os
# import sys
import json
import time
import copy
import torch
import click
import numpy as np
import warnings

from pathlib import Path
from skimage.io import imread
from losses_pytorch.detection import weighted_hausdorff_distance_with_probs_loss

from netutils.metrics.detection import recall_score, precision_score

from precode.net_utils import ( batch_ids_generator,
                                torch_long, torch_float )
from precode.net_utils.tiling import tile_iterator
from precode.models import BoostedUnetSm as UnetSm
from precode.utils import init_determenistic, init_logging, unchain
from precode.labels.spikelet_detection import DETECTION_LABELS

warnings.simplefilter("ignore")

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
    
    for key in ['train', 'val']:
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
                    ntree=1)

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


def get_central_points(mask):
    contours, _ = cv2.findContours( mask,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE )

    central_points = list()

    for contour in contours:
        try:
            moments = cv2.moments(contour)

            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])

            central_points.append(np.array([cy, cx]))
        except:
            pass

    return np.array(central_points)

def inner_train_loop(model, opt, shed, imgpathes, maskpathes):
    model.train()
    size = len(maskpathes)

    batch_losses = list()
    
    step = 0 

    for batch_ids in config['VERBOSER'](batch_ids_generator(size, config['BATCH_SIZE'], True)):
        imgpathes_batch = imgpathes[batch_ids]
        maskpathes_batch = maskpathes[batch_ids]

        imgs_batch, masks_batch = augmented_load(imgpathes_batch, maskpathes_batch, config['TRAIN_AUG'])

        imgs_batch = torch_float(imgs_batch, config['DEVICE'])
        #masks_batch = torch_float(masks_batch, config['DEVICE'])

        points_batch = list()

        for mask in masks_batch:
            points_batch.append(get_central_points(mask))
            
#         points_batch = np.array(points_batch)
#         points_batch = torch.from_numpy(points_batch)

#         points_batch = points_batch.to(config['DEVICE'])

        logitses_batch = model(imgs_batch)

        losses = ()
        
        for logits_batch in logitses_batch:
            prob_masks_batch = torch.softmax(logits_batch, axis=1)[:, 1]
            loss = weighted_hausdorff_distance_with_probs_loss(points_batch, prob_masks_batch, eps=1e-3, alpha=-11)
#             loss = torch.nn.functional.cross_entropy(logits_batch, masks_batch)
            losses = (*losses, loss)

        losses = torch.stack(losses)
        loss = torch.mean(losses)

        loss.backward()

#         for p in model.parameters():
#             p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

        step += 1

        if step == config['ACCUMULATION_STEP']:
            opt.step()
            # shed.step(loss)
            model.zero_grad()

            step = step % config['ACCUMULATION_STEP']


        batch_losses.append(loss.item())
        assert not np.isnan(batch_losses[-1])

        del imgpathes_batch, maskpathes_batch, imgs_batch, masks_batch

    return np.mean(batch_losses)

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

def inner_val_loop(model, imgpathes, maskpathes):
    model.eval()

    precisions = list()
    recalls = list()
    f1s = list()

    IDX = 0

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
#             from skimage.io import imsave

#             colored_prob_mask = np.zeros((*mask.shape, 3))
#             colored_prob_mask[:, :, 0] = 255 * prob_mask
#             colored_prob_mask[:, :, 2] = 255 * (1 - prob_mask)
#             colored_prob_mask = colored_prob_mask.astype(np.uint8)

#             imsave(f'tmp/pred_mask_{IDX}.png', 255 * pred_mask)
#             imsave(f'tmp/prob_mask_{IDX}.png', colored_prob_mask)
#             imsave(f'tmp/mask_{IDX}.png', 255 * mask)

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

    return score, ({
        'f1s': np.mean(f1s),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls)
    })

import cv2

def fit(model, data):
    train_losses = list()
    val_scores = list()

    model.to(config['DEVICE'])

    opt = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
#     shed = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.1, last_epoch=-1, verbose=False)
    # shed = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max')
    shed = None

    epochs_without_going_up = 0
    best_score = 0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(config['EPOCHS']):
        start_time = time.perf_counter()

        loss = inner_train_loop( model,
                                 opt,
                                 shed,
                                 *data['train'] )

        config['LOGGER'].info(f'epoch - {epoch+1} loss - {loss:.6f}')
        train_losses.append(loss)

        score, additional = inner_val_loop( model,
                                            *data['val'] )

        val_scores.append(score)
        config['LOGGER'].info(f'epoch - {epoch+1} score - {100 * score:.2f}%')

        for key in additional:
            config['LOGGER'].info(f'epoch - {epoch+1} {key} - {100 * additional[key]:.2f}%')

        if best_score < score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_going_up = 0
            best_epoch = epoch + 1
        else:
            epochs_without_going_up += 1

        if epochs_without_going_up == config['STOP_EPOCHS']:
            break

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        config['LOGGER'].info(f'elapsed time {elapsed_time:.2f} s')

        if config['DEBUG']:
            debugging_info['epoch_loss_trace'].append(round(loss, 3))
            debugging_info['epoch_score_trace'].append(round(100*score, 3))
            debugging_info['epoch_times'].append(round(elapsed_time, 3))

            for key in additional:
                additional[key] = round(100*additional[key], 3)

            debugging_info['epoch_additional_score_trace'].append(additional)

    config['LOGGER'].info(f"Best model saved on {best_epoch} epoch")
    model.load_state_dict(best_state)

def store(model):
    state = model.state_dict()

    torch.save(state, config['PREFIX'] + config['MODELNAME'])

def store_debug():
    if not config['DEBUG']:
        return

    if torch.cuda.is_available():       
        debugging_info['max_memory_consumption'] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    else:
        pass

    with open(config['PREFIX'] + config['DEBUGNAME'], 'w') as f:
        json.dump(debugging_info, f)

def init_labels():
    config['LABEL_MAP'] = DETECTION_LABELS

    return len(config['LABEL_MAP'])

@click.command()
@click.option('--splitpath', '-sp', type=str, default='split.json')
@click.option('--backbone', '-bone', type=str, default='efficientnet-b0')
@click.option('--modelname', '-mn', type=str, default='ribera.bin')
@click.option('--epochs', '-e', type=int, default=200)
@click.option('--batch_size', '-bs', type=int, default=16)
@click.option('--stop_epochs', '-se', type=int, default=200)
@click.option('--learning_rate', '-lr', type=float, default=10e-4)
@click.option('--verbose', is_flag=True)
@click.option('--debugname', '-dn', type=str, default='ribera.json')
@click.option('--debug', is_flag=True)
@click.option('--accumulation_step', '-as', type=int, default=1)
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

    config['LOGGER'].info(f'fit model')
    fit(model, data)

    config['LOGGER'].info(f'store model')
    store(model)

    store_debug()

if __name__ == '__main__':
    main()
