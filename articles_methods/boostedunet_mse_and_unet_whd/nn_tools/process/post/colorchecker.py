import numpy as np

from cc3d import connected_components
from skimage.morphology import remove_small_holes

def apply_FP_FN_filtration_colorchecker(mask):
    """
        NOTE:
            assumed that 
                1. ColorChecker is one in the image
                2. mask is binary
    """

    components, ncomponents = connected_components(mask, connectivity=4, return_N=True)
    sizes = [ np.sum(components == idx + 1) for idx in np.arange(ncomponents) ]

    mcc_mask = np.array(mask)
    mcc_mask[components != (np.argmax(sizes) + 1)] = 0

    area_threshold = np.sum(mcc_mask) // 4
    mcc_mask = remove_small_holes(mcc_mask, area_threshold=area_threshold)

    return mcc_mask


