import os
import cv2
import albumentations as A

from pathlib import Path


train_aug = A.Compose([ A.HorizontalFlip( p=0.5 ),
                        A.ShiftScaleRotate( shift_limit=0.0,
                                            scale_limit=(0. , 0.1),
                                            rotate_limit=5,
                                            interpolation=cv2.INTER_LINEAR,
                                            p=0.75 ),
                        A.ColorJitter ( brightness=0.4,
                                        contrast=0.4,
                                        saturation=0.,
                                        hue=0.,
                                        always_apply=False,
                                        p=0.75 ),
#                         A.Resize ( height=2048,
#                                    width=2048,
#                                    interpolation=1,
#                                    always_apply=False,
#                                    p=1. ),
                        # A.FDA ( reference_images=[
                        #     str(name) for name in (Path(os.getenv('PLOIDY_WORKSPACE')) / 'data' / 'segmentaion' /'fda' / 'black_owns' ).glob('*.jpg')
                        # ] + [
                        #     str(name) for name in (Path(os.getenv('PLOIDY_WORKSPACE')) / 'data' / 'segmentaion' /'fda' / 'scanner_protocol' ).glob('*.jpg')
                        # ]
                        # ),
                        A.Normalize ( mean=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225),
                                      max_pixel_value=255.0,
                                      always_apply=False,
                                      p=1.0 )
                  ])

class Eclipce(A.ImageOnlyTransform):
    import numpy as np
    def apply(self, img, **params):
        img = img.astype(int)
        mask = img > 0

        img[mask] = img[mask] - np.random.randint(0, 100)

        return img.clip(0, 255)
    
    def get_transform_init_args_names(self):
        return {}

awns_aug = A.Compose([ Eclipce ( p=0.75 ),
                  ])

infer_aug = A.Compose([ 
#     A.Resize ( height=2048,
#                                    width=2048,
#                                    interpolation=1,
#                                    always_apply=False,
#                                    p=1. ),
                        A.Normalize ( mean=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225),
                                      max_pixel_value=255.0,
                                      always_apply=False,
                                      p=1.0 )
                      ])
