import os
import torch
import logging
import warnings
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp


from skimage import io
from pathlib import Path
from skimage.transform import resize


from utils.measure_time import measure_time
from utils.normalise_distance import detection_score
from preprocessing.create_dataloaders import create_dataloaders
from preprocessing.augmentation_transformations import get_transformations

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

ROUND_LIMIT = 6

@measure_time
def testing(config):

    test_transformations = get_transformations(config, "test")

    test_dataloader = create_dataloaders(config, "test", test_transformations)

    precisions = list()
    recalls = list()
    f1s = list()

    logger.info("Testing model started...")

    model = config["ARCHITECTURE"]
    model.load_state_dict(torch.load(config["MODEL_NAME"]))
    model.eval()
    with torch.no_grad():
        for i, (img, mask) in enumerate(test_dataloader):
            img = img.to(config["DEVICE"])
            mask = mask.to(config["DEVICE"]).unsqueeze(1)

            preds = torch.sigmoid(model(img))

            masks = (mask > 0.5).float()
            preds = (preds > 0.5).float()

            limit = config["BATCH_SIZE"]
            if i == (len(config["DATA_ORG"]["test"]) // config["BATCH_SIZE"]):
                limit = (len(config["DATA_ORG"]["test"]) % config["BATCH_SIZE"])
            for k in range(limit):
                pred_mask = resize(preds[k][0].cpu().data.numpy(), (config["CROP_SIZE"], config["CROP_SIZE"])) 
                pred_mask = pred_mask.astype(np.uint8)

                true_mask = resize(masks[k][0].cpu().data.numpy(), (config["CROP_SIZE"], config["CROP_SIZE"]))
                true_mask = true_mask.astype(np.uint8)

                io.imsave(os.path.join(config["SAVE_OUTPUT_PATH"], config["EXPERIMENT_NAME"], config["DATA_ORG"]["test"][i * config["BATCH_SIZE"] + k] + '_true.png'), true_mask * 255)
                io.imsave(os.path.join(config["SAVE_OUTPUT_PATH"], config["EXPERIMENT_NAME"], config["DATA_ORG"]["test"][i * config["BATCH_SIZE"] + k] + '_pred.png'), pred_mask * 255)

                name = str(Path(config["DATA_ORG"]["test"][i]).stem)

                distance = config["RADIUS"] * np.sqrt(config["COEFS"][name]) / config["RATIOS"][name]

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
                # print(precision, recall, fscore)

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(fscore)

        logger.info("Metrics on test")
        logger.info(f"Precision: {round(np.mean(precisions), ROUND_LIMIT)}")
        logger.info(f"Recall: {round(np.mean(recalls), ROUND_LIMIT)}")
        logger.info(f"F-score: {round(np.mean(f1s), ROUND_LIMIT)}")