import numpy as np

import os
import cv2
import json
import warnings

warnings.simplefilter("ignore")

from skimage import draw

from pathlib import Path
from skimage import io
from tqdm import tqdm

from skimage import draw
from skimage.transform import resize
from skimage.morphology import binary_dilation
from matplotlib import pyplot as plt

SIZE = 384

coefs = {}
ratios = {}
equal = {}

# IMG_PATH = "../../../all_right_data_dont_change/images/"
# CIRCLE_PATH = "../../../all_right_data_dont_change/circles/"

IMG_PATH = "input/images/"
CIRCLE_PATH = "input/circles/"

OUT_IMG = "output/images/"
OUT_CIRCLE = "output/circles/"

old_coefs = {}
with open("coefs.json") as f:
    old_coefs = json.load(f)

old_ratios = {}
with open("ratios.json") as f:
    old_ratios = json.load(f)

old_equal = {}
with open("equal.json") as f:
    old_equal = json.load(f)

equal["train"] = []
equal["val"] = []
equal["test"] = []

for key in ["train", "val", "test"]:
    for i in old_equal[key]:
        name = i[:-4]
        coefs[name] = old_coefs[name]
        ratios[name] = old_ratios[name]
        equal[key].append(name + '.jpg')

        img = io.imread(CIRCLE_PATH + name + '_spmk_1.0.png')

        xc, yc = np.where(img > 0)

        x_center_coord = np.min(xc) + (np.max(xc) - np.min(xc)) // 2
        y_center_coord = np.min(yc) + (np.max(yc) - np.min(yc)) // 2

        x1 = abs(x_center_coord - np.min(xc))
        x2 = abs(x_center_coord - np.max(xc))
        y1 = abs(y_center_coord - np.min(yc))
        y2 = abs(y_center_coord - np.max(yc))

        crop_size = np.max([x1, x2, y1, y2])

        min_img_shape_half = np.min([img.shape[0], img.shape[1]]) // 2
        margin_max = min_img_shape_half - crop_size - 1
        margin_min = int(0.01 * (min_img_shape_half * 2))

        PARTS_NUM = 3
        part_of_delta_margin = (margin_max - margin_min) // PARTS_NUM

        for i in range(PARTS_NUM):
            margin_size = part_of_delta_margin * (i + 1)
            percentage_of_margin = round(margin_size / (min_img_shape_half * 2), 2)
            # print("percentage_of_margin", percentage_of_margin)

            side_size = crop_size + margin_size

            xs = x_center_coord - side_size
            xe = x_center_coord + side_size
            ys = y_center_coord - side_size
            ye = y_center_coord + side_size

            if xs < 0:
                xe = xe + abs(xs)
                xs = 0
            elif xe > img.shape[0]:
                xs = xs - abs(xe - img.shape[0])
                xe = img.shape[0]

            if ys < 0:
                ye = ye + abs(ys)
                ys = 0
            elif ye > img.shape[1]:
                ys = ys - abs(ye - img.shape[1])
                ye = img.shape[1]

            img = io.imread(CIRCLE_PATH + name + '_spmk_1.0.png')
            cropped = img[xs:xe, ys:ye]
            # print(xs, xe, ys, ye)
            # print("shape", cropped.shape[0], cropped.shape[1])
            # print("\n")
            # print(name)
            # print(xs, xe, ys, ye)

            cropped[cropped > 0] = 255

            contours = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0] # cv2.CHAIN_APPROX_SIMPLE
            spikelets_centers = []
            for c in contours:
                M = cv2.moments(c)
                # центр масс
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                spikelets_centers.append([cy, cx])

            spikelets_centers = sorted(spikelets_centers, key=lambda p: p[0])
            mm_masks = np.zeros_like(cropped)
            radius = 1 * np.sqrt(old_coefs[name])
            for center in spikelets_centers:
                rr, cc = draw.disk(center=tuple(center), radius=radius, shape=mm_masks.shape)
                mm_masks[rr, cc] = 1
            
            cropped = cv2.resize(mm_masks, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            cropped[cropped > 0] = 255
            cropped = cropped.astype(np.uint8)
            ##########################################

            io.imsave(OUT_CIRCLE + name + '_' + str(percentage_of_margin) + '.png', cropped)

            if name[:5] == "Spike":
                img = io.imread(IMG_PATH + name + '.JPG')
            else:
                img = io.imread(IMG_PATH + name + '.jpg')
                
            cropped = img[xs:xe, ys:ye]

            coefs[name + '_' + str(percentage_of_margin)] = old_coefs[name]
            ratios[name + '_' + str(percentage_of_margin)] = (xe - xs) / SIZE
            equal[key].append(name + '_' + str(percentage_of_margin) + '.jpg')

            cropped = resize(cropped, (SIZE, SIZE)) * 255
            cropped = cropped.astype(np.uint8)
            io.imsave(OUT_IMG + name + '_' + str(percentage_of_margin) + '.jpg', cropped)

with open('coefs_2.json', 'w') as f:
    json.dump(coefs, f)

with open('ratios_2.json', 'w') as f:
    json.dump(ratios, f)

with open('equal_2.json', 'w') as f:
    json.dump(equal, f)