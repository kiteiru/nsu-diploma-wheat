import albumentations as A

from albumentations.pytorch import ToTensorV2


def get_transformations(config, set_name):
    if set_name == "train":
        return A.Compose([A.Resize(height=config["CROP_SIZE"], width=config["CROP_SIZE"]),
                          A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=config["COLOR_JITTER_PROBABILITY"]),
                          A.HorizontalFlip(p=config["HORIZONTAL_FLIP_PROBABILITY"]),
                          A.VerticalFlip(p=config["VERTICAL_FLIP_PROBABILITY"]),
                          A.Rotate(limit=30, p=config["ROTATE_PROBABILITY"]),
                          # A.ShiftScaleRotate(shift_limit=0.4, scale_limit=0.0, rotate_limit=0, p=0.2),


                        #   A.HorizontalFlip(p=config["HORIZONTAL_FLIP_PROBABILITY"]),
                        #   A.VerticalFlip(p=config["VERTICAL_FLIP_PROBABILITY"]),
                        #   A.ShiftScaleRotate(shift_limit=1.0, scale_limit=0.0, rotate_limit=90, p=config["SHIFT_ROTATE_PROBABILITY"]),
                        #   A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=config["COLOR_JITTER_PROBABILITY"]),
                          # A.Downscale(scale_min=0.25, scale_max=0.75, p=config["DOWNSCALE_PROBABILITY"]),
                          # A.ISONoise(p=config["ISO_NOISE_PROBABILITY"]),
                          # A.GaussianBlur(p=config["GAUSSIAN_BLUR_PROBABILITY"]),
                          # A.MotionBlur(blur_limit=5, p=config["MOTION_BLUR_PROBABILITY"]),
                        #   A.RGBShift(p=config["RGB_SHIFT_PROBABILITY"]),
                          ToTensorV2()])
    else:
        return A.Compose([A.Resize(height=config["CROP_SIZE"], width=config["CROP_SIZE"]),
                          ToTensorV2()])