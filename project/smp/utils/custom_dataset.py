import os
import fnmatch

from skimage import io
from torch.utils.data import Dataset

class SpikeletsDataset(Dataset):
    def __init__(self, DIRECTORY, transform=None):
        self.dir = DIRECTORY
        self.transform = transform
        self.images = fnmatch.filter(os.listdir(self.dir + "/images/"), '*.[jJ][pP][gG]')
        self.masks = fnmatch.filter(os.listdir(self.dir + "/masks/"), '*.[jJ][pP][gG]')

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, "images", self.images[index])
        mask_path = os.path.join(self.dir, "masks", self.masks[index])

        img = io.imread(img_path)
        mask = io.imread(mask_path)
        mask[mask == 255.0] = 1.0

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return img, mask

    def __len__(self):
        return len(self.images)
