import os
import numpy as np
import json

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class SpikeletsDataset(Dataset):
    def __init__(self, data_org, img_path, mask_path, set_name, transformations):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transformations = transformations

        self.images = [(str(Path(i).stem) + ".jpg") for i in data_org[set_name]]
        self.masks = [(str(Path(i).stem) + ".png") for i in data_org[set_name]]

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.images[index])
        mask_path = os.path.join(self.mask_path, self.masks[index])

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255
        
        if self.transformations is not None:
            augmentations = self.transformations(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask
    
    def __len__(self):
        return len(self.images)