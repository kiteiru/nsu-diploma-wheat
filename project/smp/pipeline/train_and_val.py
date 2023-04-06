import os
import cv2
import sys
import json
import shutil
import fnmatch
import logging
import numpy as np
import torch
import torch.nn.functional as F
#import utils.get_args as get_args
import segmentation_models_pytorch as smp
import torchvision
import albumentations as A  
import pandas as pd

sys.path.append("..")

from test_model import testing
from utils.measure_time import measure_time
from utils.spikelets_dataset import SpikeletsDataset

from pathlib import Path
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from utils.prepare_dataloader import prepare_dataloader
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from skimage import io
from skimage.transform import resize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

DATA_ORG = sys.argv[1]
GEOMETRY = sys.argv[2]
NAME = DATA_ORG + "_" + GEOMETRY

TRAIN_INP_DIR = '../../../all_cropped_images/'
TRAIN_OUT_DIR = '../../../all_cropped_' + GEOMETRY + '/'

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-3
BATCH_SIZE    = 16
NUM_EPOCHS    = 5
IMAGE_HEIGHT  = 384
IMAGE_WIDTH   = 384

ENCODER = 'efficientnet-b3'

os.makedirs(os.path.join("..", "results", NAME))

list_precision = []
list_recall = []
list_fscore = []
list_accuracy = []

def training(loader, model, optimizer, loss_fn, scheduler):
    loop = tqdm(loader)

    model.train()
    for batch_idx, (image, mask) in enumerate(loop):
        image = image.to(device=DEVICE)
        mask = mask.float().unsqueeze(1).to(device=DEVICE)

        predictions = model(image)
        loss = loss_fn(predictions, mask)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # scheduler.step(loss.item())

        loop.set_postfix(loss=loss.item())

def validating(loader, model, device="cuda"):
    # change on normalised circles metric
    num_pixels = 0
    dice_score = 0

    score_f1 = 0
    score_recall = 0
    score_precision = 0
    score_acc = 0

    model.eval()

    with torch.no_grad():
        for img, mask in tqdm(loader):
            img = img.to(device)
            mask = mask.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(img))

            y_true = mask.cpu().numpy()
            y_true = y_true > 0.5
            y_true = y_true.astype(np.uint8)
            y_true = y_true.reshape(-1)

            y_pred = preds.cpu().numpy()
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.uint8)
            y_pred = y_pred.reshape(-1)

            score_f1 = f1_score(y_true, y_pred)
            score_recall = recall_score(y_true, y_pred)
            score_precision = precision_score(y_true, y_pred)
            score_acc = accuracy_score(y_true, y_pred)

            dice_score += (2 * (preds * mask).sum()) / ((preds + mask).sum() + 1e-7)

    # print(f"score_f1: {score_f1/len(loader)*100:.2f}")
    # print(f"score_recall: {score_recall/len(loader)*100:.2f}")
    # print(f"score_precision: {score_precision/len(loader)*100:.2f}")
    # print(f"score_acc: {score_acc/len(loader)*100:.2f}")

    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # fscore = (2 * precision * recall) / (precision + recall)

    print("train metrics:")
    print(f"Precision {score_precision*100:.2f}%")
    print(f"Recall {score_recall*100:.2f}%")
    print(f"F-Score {score_f1*100:.2f}%")
    print(f"Accuracy {score_acc*100:.2f}%")
    print(f"Dice score: {dice_score/len(loader)*100:.2f}")

    list_precision.append(score_precision*100)
    list_recall.append(score_recall*100)
    list_fscore.append(score_f1*100)
    list_accuracy.append(score_acc*100)


def get_loaders(inp_dir, mask_dir, batch_size, train_transform, val_transform):
    train_ds = SpikeletsDataset(DATA_ORG, input_dir=inp_dir, output_dir=mask_dir, set_name="train", transform=train_transform)
    val_ds = SpikeletsDataset(DATA_ORG, input_dir=inp_dir, output_dir=mask_dir, set_name="val", transform=val_transform)
    test_ds = SpikeletsDataset(DATA_ORG, input_dir=inp_dir, output_dir=mask_dir, set_name="test", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

@measure_time
def train_and_val_model():
    # need to transfer transform to preprocessing/augment_data.py
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.ColorJitter(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2(),
        ],
    )

    train_loader, val_loader, test_loader = get_loaders(TRAIN_INP_DIR, TRAIN_OUT_DIR, BATCH_SIZE, train_transform, val_transform)
    # inputs, masks = next(iter(train_loader))

    model = smp.Unet(encoder_name=ENCODER, in_channels=3, classes=1, activation=None).to(DEVICE)
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # validating(val_loader, model, device=DEVICE)

    for epoch in range(NUM_EPOCHS):

        print('########################## epoch: '+str(epoch))
        training(train_loader, model, optimizer, loss_fn, scheduler)
        
        validating(val_loader, model, device=DEVICE)

        # if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join("..", "results", "models", NAME + f"_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved on {epoch + 1}, path is {checkpoint_path}")


    data = {'Precision': list_precision, 
            'Recall': list_recall,
            'Fscore': list_fscore, 
            'Accuracy': list_accuracy}

    df = pd.DataFrame(data)

    df.to_csv(os.path.join("..", "results", NAME + ".csv"), sep=';')

    testing(test_loader, model, DEVICE, BATCH_SIZE, NAME)