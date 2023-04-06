
import os, time
import numpy as np
import cv2
import json
import imageio
import torch
import torchvision
import segmentation_models_pytorch as smp
import albumentations as A
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from utils.measure_time import measure_time

from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from skimage import io
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from skimage.transform import resize
from pathlib import Path

@measure_time
def testing(loader, model, device, batch_size, name):
    # change on normalised circles metric
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    num_pixels  = 0
    dice_score  = 0

    score_f1 = 0
    score_recall = 0
    score_precision = 0
    score_acc = 0

    model.eval()

    with torch.no_grad():
        for i, (img, mask) in tqdm(enumerate(loader)):
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

            output = (preds > 0.5).float()

            my_dpi = 30

            limit = batch_size
            if i == 21:
                limit = 12
            for k in range(limit):
                image = resize(output[k][0].cpu().data.numpy(), (384, 384)) * 255
                image = image.astype(np.uint8)

                io.imsave(os.path.join("..", "results", name, 'output' + str(i) + "_" + str(k) + '.png'), image)

        print("test metrics:")
        print(f"Precision {score_precision*100:.2f}%")
        print(f"Recall {score_recall*100:.2f}%")
        print(f"F-Score {score_f1*100:.2f}%")
        print(f"Accuracy {score_acc*100:.2f}%")
        print(f"Dice score: {dice_score/len(loader)*100:.2f}")