import os
import monai
import torch
import warnings
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as A
import segmentation_models_pytorch as smp

from skimage import io
from albumentations.pytorch import ToTensorV2
from monai.inferers import sliding_window_inference

warnings.filterwarnings("ignore")

def inference(img_dir, model_path):
    print("Inference model started...")
    for img_path in Path(img_dir).glob('*.jpg'):

        name = str(Path(img_path).stem)
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255
        infer_transformations = A.Compose([A.Resize(384, 384), ToTensorV2()])
        augmentations = infer_transformations(image=img)
        img = augmentations["image"]
        img = img.unsqueeze(0)
        img = img.to(device="cuda", dtype=torch.float)

        model = smp.Unet(encoder_name='efficientnet-b4',
                        in_channels=3,
                        classes=1,
                        activation=None).to(device="cuda")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            pred_mask = torch.sigmoid(model(img))
            pred_mask = (pred_mask > 0.5).float() * 255
            pred_mask = pred_mask[0][0].cpu().data.numpy()
            pred_mask = pred_mask.squeeze().astype(np.uint8)
        # pred_img = Image.fromarray(pred_mask)
        io.imsave(os.path.join("infer", "infer_" + name + ".jpg"), pred_mask)


if __name__ == "__main__":
    IMG_DIR = "infer"
    MODEL_PATH = "../results/models/circles_equal_unet_efficientnet-b4_bcelogits_2mm_15-07-24:23-04/best_on_42_epoch.pt"
    inference(IMG_DIR, MODEL_PATH)