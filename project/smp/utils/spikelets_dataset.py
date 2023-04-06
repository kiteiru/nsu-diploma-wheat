import os
import numpy as np
import json

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class SpikeletsDataset(Dataset):
    def __init__(self, data_org, input_dir, output_dir, set_name, transform=None):
        self.data_org = data_org
        self.input_dir  = input_dir
        self.output_dir = output_dir
        self.transform  = transform

        data = None

        with open('../data_organization/' + self.data_org + '.json', 'r') as f:
            data = json.load(f)

        self.images = [(str(Path(i).stem) + ".jpg") for i in data[set_name]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.input_dir, self.images[index])
        mask_path = os.path.join(self.output_dir, self.images[index])
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255
        
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]
        
        return img, mask