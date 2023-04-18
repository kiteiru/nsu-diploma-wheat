import os
import sys
import cv2
import pandas as pd

from flask import Flask, redirect, jsonify, request, url_for, render_template, flash, abort, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

from config import DevConfig

app = Flask(__name__)

devConfig = DevConfig()
app.config["UPLOADS"] = devConfig.UPLOADS
app.config["CLIENT_IMAGES"] = devConfig.CLIENT_IMAGES

data = {"Name": [],
        "Spikelets Num": []}

allowed_exts = {'jpg', 'jpeg', 'png'}

def check_allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

def save_in_buffer(img, pred_img):
    encoded_strings = []
    for pic in [img, pred_img]:
        with BytesIO() as buf:
            pic.save(buf, 'png')
            image_bytes = buf.getvalue()
        encoded_strings.append(base64.b64encode(image_bytes).decode())
    return tuple(encoded_strings)

def count_spikelets(img):
    img = np.array(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def predict(img):
    img = np.array(img.convert("RGB"), dtype=np.float32) / 255
    infer_transformations = A.Compose([A.Resize(384, 384), ToTensorV2()])
    augmentations = infer_transformations(image=img)
    img = augmentations["image"]
    img = img.unsqueeze(0)
    img = img.to(device="cuda", dtype=torch.float)

    model = smp.Unet(encoder_name='efficientnet-b0',
                     in_channels=3,
                     classes=1,
                     activation=None).to(device="cuda")
    MODEL_PATH = "../smp/results/models/circles_equal_unet_efficientnet-b0_bce_with_logits_2mm_18-53-48:18-04/best_on_87_epoch.pt"
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        pred_mask = torch.sigmoid(model(img))
        pred_mask = (pred_mask > 0.5).float() * 255
        pred_mask = pred_mask[0][0].cpu().data.numpy()
        pred_mask = pred_mask.squeeze().astype(np.uint8)
    pred_img = Image.fromarray(pred_mask)

    return pred_img


@app.route("/",methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                print('No file attached in request')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                print('No file selected')
                return redirect(request.url)
            if file and check_allowed_file(file.filename):
                filename = str(Path(secure_filename(file.filename)).stem)
                
                img = Image.open(file.stream)
                pred_img = predict(img)

                spikelets_num = count_spikelets(pred_img)

                data["Name"].append(filename)
                data["Spikelets Num"].append(spikelets_num)

                encoded_img, encoded_pred_img = save_in_buffer(img, pred_img)

                df = pd.DataFrame(data, index=None)
                print(df)
                

            return render_template('index.html', img_data=encoded_img, out_data=encoded_pred_img, spikelets_num=spikelets_num, tables=[df.to_html(classes='data')], titles=df.columns.values)
        except FileNotFoundError:
            abort(404)
    else:
        return render_template('index.html', img_data="", out_data="", spikelets_num="", tables=[], titles="")
    

if __name__ == "__main__":
    app.run()