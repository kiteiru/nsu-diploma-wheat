import albumentations as A

from albumentations.pytorch import ToTensorV2


def get_transformations(config, set_name):
    if set_name == "train":
        return A.Compose([A.Resize(height=config["CROP_SIZE"], width=config["CROP_SIZE"]),
                          A.ColorJitter(p=0.2),
                          A.HorizontalFlip(p=0.5),
                          A.VerticalFlip(p=0.5),
                          A.Rotate(limit=30, p=0.5),
                          ToTensorV2()])
    else:
        return A.Compose([A.Resize(height=config["CROP_SIZE"], width=config["CROP_SIZE"]),
                          ToTensorV2()])