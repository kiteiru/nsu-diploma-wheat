import albumentations as A
import warnings

from skimage import io
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings("ignore")

# TODO reformat this func
def augment_data(config):
    # IMAGE_DIR = "../cropped_images/512/british_images/"
    # MASK_DIR = "../cropped_images/512/british_circle_masks/"

    IMAGE_DIR = config["CROP_DIR"] + "/british_images/"
    MASK_DIR = config["CROP_DIR"] + "/british_circle_masks/"

    IMAGE_AUG_DIR = config["DATA_DIR"] + "/images/"
    MASK_AUG_DIR = config["AUG_DIR"] + "/masks/"
        
    aug = A.Compose([
        A.VerticalFlip(p = 0.8), 
        A.HorizontalFlip(p = 0.8), 
        # A.GridDistortion(p = 0.6, distort_limit = 0.4),
        A.RandomContrast(p = 0.9, limit = 0.5),
        A.ShiftScaleRotate(p = 0.7, shift_limit = 0.1, scale_limit = 0.4, rotate_limit = 90)
    ])

    # number of augmented data = number of input data * DATA_BOOST
    for path in tqdm([*(Path(IMAGE_DIR)).glob('*.[jJ][pP][gG]')]):
        for j in range(config["DATA_BOOST"]):
            image = str(path)
            mask = MASK_DIR + str(path.stem) + ".jpg"
            
            orig_image = io.imread(image)
            orig_mask = io.imread(mask)
            
            augmented = aug(image = orig_image, mask = orig_mask)
            aug_image = augmented["image"]
            aug_mask = augmented["mask"]
            
            io.imsave(IMAGE_AUG_DIR + "aug_img_" + str(path.stem) + "_" + str(j) + ".jpg", aug_image)
            io.imsave(MASK_AUG_DIR + "aug_mask_" + str(path.stem) + "_" + str(j) + ".jpg", aug_mask)