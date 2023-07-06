import os
import cv2
import torch
import warnings
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
import time
import argparse
import pandas as pd

from PIL import Image
from pathlib import Path
from skimage import io
from albumentations.pytorch import ToTensorV2
# from monai.inferers import sliding_window_inference

warnings.filterwarnings("ignore")

data = {"Name": [],
        "Spikelets Num": []}

def count_spikelets(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def inference(input, model, output, csv_name, device):
    print("Inference model started...")
    model.eval()
    infer_transformations = A.Compose([A.Resize(384, 384), ToTensorV2()])
    
    start = time.time()
    for img_path in Path(input).glob('*.jpg'):

        name = str(Path(img_path).stem)
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255
        augmentations = infer_transformations(image=img)
        img = augmentations["image"]
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float)

        with torch.no_grad():
            pred_mask = torch.sigmoid(model(img))
            pred_mask = (pred_mask > 0.5).float() * 255
            pred_mask = pred_mask[0][0].cpu().data.numpy()
            pred_mask = pred_mask.squeeze().astype(np.uint8)
        # pred_img = Image.fromarray(pred_mask)
        io.imsave(os.path.join(output, "infer_" + name + ".jpg"), pred_mask)
        spikelets_num = count_spikelets(pred_mask)

        data["Name"].append(name)
        data["Spikelets Num"].append(spikelets_num)

    df = pd.DataFrame(data, index=None)
    df.to_csv(csv_name + '.csv', index=False)

    end = time.time()
    print(f"{round((end - start), 7)} seconds elapsed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdata", "-in", type=str, default="inference_input", help="directory path with input data for inference")
    parser.add_argument("--modelpath", "-model", type=str, default="../results/models/circles_equal_unet_efficientnet-b4_bce_2mm_03-40-55:08-05/best_on_257_epoch.pt", help="model path")
    parser.add_argument("--outputdata", "-out", type=str, default="inference_output", help="directory path with further output after inference")
    parser.add_argument("--csvfile", "-csv", type=str, default="inference", help="csv filename")
    args = parser.parse_args()

    IN_DIR = args.inputdata
    MODEL_PATH = args.modelpath
    OUT_DIR = args.outputdata
    CSV_NAME = args.csvfile
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = smp.Unet(encoder_name='efficientnet-b4',
                    in_channels=3,
                    classes=1,
                    activation=None).to(device=DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    
    inference(IN_DIR, model, OUT_DIR, CSV_NAME, DEVICE)