import numpy as np

from functools import partial
from multiprocessing import Pool
from collections import defaultdict

def __apply_image_augmentations_without_parallelization(imgs, masks, aug):
    auged_imgs = list()
    auged_masks = list()

    for img, mask in zip(imgs, masks):
        auged = aug(image=np.asarray(img), mask=np.asarray(mask))

        aug_img = auged['image']
        aug_mask = auged['mask']

        auged_imgs.append(aug_img)
        auged_masks.append(aug_mask)

    auged_imgs = np.array(auged_imgs)
    auged_imgs = np.moveaxis(auged_imgs, -1, 1)

    auged_masks = np.array(auged_masks, dtype=np.uint8)

    return auged_imgs, auged_masks

def __apply_image_augmentations_loop(args, aug):
    img, mask = args
    auged = aug(image=np.asarray(img), mask=np.asarray(mask))

    return auged['image'], auged['mask']

def __apply_image_augmentations_with_parallelization(imgs, masks, aug, *, njobs=1):
    args = zip(imgs, masks)
    loop = partial(__apply_image_augmentations_loop, aug=aug)

    pool = Pool(njobs)
    auged_imgs, auged_masks = zip(*pool.map(loop, args))

    auged_imgs = np.array(auged_imgs)
    auged_imgs = np.moveaxis(auged_imgs, -1, 1)

    auged_masks = np.array(auged_masks, dtype=np.uint8)

    return auged_imgs, auged_masks

def apply_image_augmentations(imgs, masks, aug, *, njobs=1):
    if njobs == 1:
        return __apply_image_augmentations_without_parallelization(imgs, masks, aug)
    else:
        return __apply_image_augmentations_with_parallelization(imgs, masks, aug, njobs=njobs)

class AlbumentationsMaskKeywordIgnorer(object):
    def __init__(self, aug):
        self.__aug = aug

    def add_targets(self, *args, **kwargs):
        return self.__aug.add_targets(*args, **kwargs)

    def __call__(self, **kwargs):
        filtred_kwargs = dict(kwargs)

        for key in kwargs.keys():
            if key.startswith('mask'):
                del filtred_kwargs[key]

        auged = defaultdict(lambda *args, **kwargs: 0)
        auged.update(self.__aug(**filtred_kwargs))

        return auged

class NoneSubscriptable(object):
    def __getitem__(self, *args, **kwargs):
        return None

def InfiniteNoneSubscriptableGenerator():
    while True:
        yield NoneSubscriptable()

def apply_only_image_augmentations(imgs, aug):
    masks = InfiniteNoneSubscriptableGenerator()
    aug = AlbumentationsMaskKeywordIgnorer(aug)

    auged_imgs, _ = apply_image_augmentations(imgs, masks, aug)

    return auged_imgs
