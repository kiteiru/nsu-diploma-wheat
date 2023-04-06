import os
import cv2
import sys
import json
import torch
import shutil
import fnmatch
import logging
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from pathlib import Path
from utils.custom_dataset import SpikeletsDataset
from torch.utils.data import TensorDataset, DataLoader

def get_original_number_of_data(data_organization):
    num = 0
    for key in list(data_organization.keys()):
        num += len(data_organization[key])
    return num

def split_data_by_folders(config):
    data_organization = None

    with open(config["DATA_JSON"], 'r') as f:
        data_organization = json.load(f)

    if config["DATA_BOOST"] is None:
        num_of_aug_data = len(fnmatch.filter(os.listdir(config["DATA_DIR"] + "/images/"), '*.[jJ][pP][gG]'))
        repeats = int(num_of_aug_data / get_original_number_of_data(data_organization))
    else:
        repeats = config["DATA_BOOST"]

    for set_name in tqdm(list(data_organization.keys())):
        for path in tqdm(data_organization[set_name]):
            for i in range(repeats):
                path = Path(path)

                name = str(path.stem) + "_" + str(i) + ".jpg"
                shutil.copy(os.path.join(config["DATA_DIR"], "images", name),
                            os.path.join(config["SPLIT_DIR"], set_name, "images", name))
                shutil.copy(os.path.join(config["DATA_DIR"], "masks", name),
                            os.path.join(config["SPLIT_DIR"], set_name, "masks", name))


def prepare_dataloader(config):

    # uncomment in future
    # split_data_by_folders(config)

    pic_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SpikeletsDataset(os.path.join(config["SPLIT_DIR"], "train"),
                                     pic_transform)
    
    val_dataset = SpikeletsDataset(os.path.join(config["SPLIT_DIR"], "val"),
                                   pic_transform)
    
    print("train_loader...")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config["BATCH_SIZE"],
                                  shuffle=True)
    print("val_loader...")
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config["BATCH_SIZE"],
                                shuffle=True)
    print("finished dataloaders")

    # train_images = []
    # train_masks = []

    # val_images = []
    # val_masks = []

    # data_organization = None

    # with open(config["DATA_JSON"], 'r') as f:
    #     data_organization = json.load(f)

    # if config["DATA_BOOST"] is None:
    #     num_of_aug_data = len(fnmatch.filter(os.listdir(config["DATA_DIR"] + "/images/"), '*.[jJ][pP][gG]'))
    #     repeats = int(num_of_aug_data / get_original_number_of_data(data_organization))
    # else:
    #     repeats = config["DATA_BOOST"]

    # for set_name in tqdm(["train", "val"]):
    #     for path in tqdm(data_organization[set_name]):
    #         for i in range(repeats):
    #             path = Path(path)
    #             img = cv2.imread(config["DATA_DIR"] + "/images/" + str(path.stem) + "_" + str(i) + ".jpg")
    #             mask = cv2.imread(config["DATA_DIR"] + "/masks/" + str(path.stem) + "_" + str(i) + ".jpg")

    #             if set_name == "train":
    #                 train_images.append(np.array(img))
    #                 train_masks.append(np.array(mask))
    #             else:
    #                 val_images.append(np.array(img))
    #                 val_masks.append(np.array(mask))

    # print("finish getting images and masks")

    return train_dataloader, val_dataloader

    # train_images = preprocess_input(train_images)
    # val_images = preprocess_input(val_images)

                

