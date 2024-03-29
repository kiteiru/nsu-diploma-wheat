import gc
import os
import copy
import random
import torch
import logging
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as A  
import segmentation_models_pytorch as smp

from torch import optim
from pathlib import Path
from skimage import io
from numpy import inf
from skimage.transform import resize

from utils.log_metrics import log_metrics
from utils.measure_time import measure_time
from utils.normalise_distance import detection_score
from preprocessing.create_dataloaders import create_dataloaders
from preprocessing.augmentation_transformations import get_transformations
# from torchsummary import summary
from pytorch_model_summary import summary

logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)

warnings.filterwarnings("ignore")

ROUND_LIMIT = 6

df_precisions = []
df_recalls = []
df_f1s = []
df_losses = []

def training(config, loader, model, optimizer, loss_fn, scheduler):
    sum_loss = 0

    model.train()
    for batch_idx, (image, mask) in enumerate(loader):
        image = image.to(device=config["DEVICE"])
        mask = mask.float().unsqueeze(1).to(device=config["DEVICE"])

        predictions = torch.sigmoid(model(image))
        loss = loss_fn(predictions, mask)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # scheduler.step(loss.item())
        sum_loss += loss.item()

    return round(sum_loss / len(loader), ROUND_LIMIT)

def validating(config, loader, model, device="cuda"):
    precisions = []
    recalls = []
    f1s = []

    model.eval()

    with torch.no_grad():
        for i, (img, mask) in enumerate(loader):
            img = img.to(device)
            mask = mask.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(img))

            masks = (mask > 0.5).float()
            preds = (preds > 0.5).float()

            limit = config["BATCH_SIZE"]
            if i == (len(config["DATA_ORG"]["val"]) // config["BATCH_SIZE"]):
                limit = (len(config["DATA_ORG"]["val"]) % config["BATCH_SIZE"])
            for k in range(limit):
                pred_mask = resize(preds[k][0].cpu().data.numpy(), (config["CROP_SIZE"], config["CROP_SIZE"]))
                pred_mask = pred_mask.astype(np.uint8)

                true_mask = resize(masks[k][0].cpu().data.numpy(), (config["CROP_SIZE"], config["CROP_SIZE"]))
                true_mask = true_mask.astype(np.uint8)

                name = str(Path(config["DATA_ORG"]["val"][i]).stem)

                distance = config["RADIUS"] * np.sqrt(config["COEFS"][name]) / (config["RATIOS"][name] * 384 / config["CROP_SIZE"])

                confusions = detection_score(name, true_mask, pred_mask, distance)
                # print(confusions)
                if confusions["TP"] + confusions["FP"] == 0:
                    continue
                precision = confusions["TP"] / (confusions["TP"] + confusions["FP"])
                recall = confusions["TP"] / (confusions["TP"] + confusions["FN"])
                
                if precision + recall == 0:
                    fscore = 0
                else:
                    fscore = 2 * (precision*recall) / (precision+recall)

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(fscore)

    mean_precision = round(np.mean(precisions), ROUND_LIMIT)
    mean_recall = round(np.mean(recalls), ROUND_LIMIT)
    mean_fscore = round(np.mean(f1s), ROUND_LIMIT)

    df_precisions.append(mean_precision)
    df_recalls.append(mean_recall)
    df_f1s.append(mean_fscore)

    logger.info("Metrics on validation")
    log_metrics(mean_precision, mean_recall, mean_fscore)

    return mean_precision, mean_recall, mean_fscore

@measure_time
def train_and_val_model(model, optimizer, loss_fn, config):

    os.makedirs(os.path.join(config["SAVE_MODEL_PATH"], config["EXPERIMENT_NAME"]), exist_ok=True)
    os.makedirs(os.path.join(config["SAVE_OUTPUT_PATH"], config["EXPERIMENT_NAME"]), exist_ok=True)

    train_transformations = get_transformations(config, "train")
    val_transformations = get_transformations(config, "val")

    train_dataloader = create_dataloaders(config, "train", train_transformations)
    val_dataloader = create_dataloaders(config, "val", val_transformations)

    
    # model = config["ARCHITECTURE"]
    # model = model.to(config["DEVICE"])

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    best_model = None
    best_epoch = 0
    best_metrics = (0, 0, 0)
    current_metrics = (0, 0, 0)

    logger.info("Training model started...")
    model_path = ""

    logger.info(summary(model, torch.zeros((8, 3, 384, 384)).to(device=config["DEVICE"]), show_hierarchical=True, show_input=True))

    # logger.info(summary(model, (3, 384, 384)))

    before_loss = inf
    for epoch in range(config["EPOCHS"]):
        logger.info(f"Current epoch num: {str(epoch + 1)}")

        after_loss = training(config, train_dataloader, model, optimizer, loss_fn, None)
        logger.info(f'Loss changed from {before_loss} to {after_loss}')
        before_loss = after_loss
        df_losses.append(after_loss)
        
        current_metrics = validating(config, val_dataloader, model, device=config["DEVICE"])

        if best_metrics[2] < current_metrics[2]:
            best_metrics = current_metrics
            best_epoch = epoch + 1
            best_model = copy.deepcopy(model.state_dict())

        # if (epoch + 1) % 10 == 0:
        #     model_path = os.path.join(config["SAVE_MODEL_PATH"], config["EXPERIMENT_NAME"], f"epoch_{epoch + 1}.pt")
        #     torch.save(model.state_dict(), model_path)
        #     logger.info(f'Model was saved on {str(epoch + 1)} epoch, path is "{model_path}"')

    model_path = os.path.join(config["SAVE_MODEL_PATH"], config["EXPERIMENT_NAME"], f"best_on_{str(best_epoch)}_epoch.pt")
    torch.save(best_model, model_path)
    logger.info(f'Best model was saved on {str(best_epoch)} epoch, path is "{model_path}"')
    log_metrics(*best_metrics)

    config["MODEL_NAME"] = model_path

    data = {'Precision': df_precisions, 
            'Recall': df_recalls,
            'Fscore': df_f1s,
            'Loss': df_losses}

    df = pd.DataFrame(data)

    df.to_csv(os.path.join(config["SAVE_MODEL_PATH"], config["EXPERIMENT_NAME"], "validation_metrics.csv"), sep=';')

    return model, optimizer, loss_fn, config, best_metrics[2]
    