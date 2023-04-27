import cv2
import albumentations as A

train_aug = A.Compose([ A.HorizontalFlip( p=0.5 ),
                        A.ShiftScaleRotate( shift_limit=0.0,
                                            scale_limit=(0. , 0.1),
                                            rotate_limit=5,
                                            interpolation=cv2.INTER_LINEAR,
                                            p=0.75 ),
                        A.OneOf([ A.ElasticTransform( alpha=1,
                                                      sigma=50,
                                                      alpha_affine=50,
                                                      interpolation=cv2.INTER_LINEAR,
                                                      border_mode=cv2.BORDER_CONSTANT,
                                                      value=0,
                                                      mask_value=0,
                                                      approximate=False,
                                                      p=0.8 ),
                                  A.GridDistortion( num_steps=4,
                                                    distort_limit=0.3,
                                                    interpolation=cv2.INTER_LINEAR,
                                                    border_mode=cv2.BORDER_CONSTANT,
                                                    p=0.8 )
                        ], p=0.9),
                        A.ColorJitter ( brightness=0.2,
                                        contrast=0.2,
                                        saturation=0.2,
                                        hue=0.2,
                                        always_apply=False,
                                        p=0.75 ),
                        A.Normalize ( mean=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225),
                                      max_pixel_value=255.0,
                                      always_apply=False,
                                      p=1.0 )
                  ]) 

infer_aug = A.Compose([ A.Normalize ( mean=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225),
                                      max_pixel_value=255.0,
                                      always_apply=False,
                                      p=1.0 )
])

