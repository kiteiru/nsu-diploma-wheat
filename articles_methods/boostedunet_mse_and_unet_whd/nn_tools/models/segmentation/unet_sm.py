import torch, torch.nn as nn
import segmentation_models_pytorch as sm

from nn_tools.models.segmentation.interface import SegmentationNet
from nn_tools.models.layer_convertors import convert_inplace, LayerConvertorSm

class UnetSm(SegmentationNet):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        self.unet = sm.Unet(in_channels=in_channels, classes=out_channels, **kwargs)
        convert_inplace(self.unet, LayerConvertorSm)

    def _forward(self, inputs):
        return self.unet(inputs)

class UnetppSm(SegmentationNet):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        self.unetpp = sm.UnetPlusPlus(in_channels=in_channels, classes=out_channels, **kwargs)
        convert_inplace(self.unetpp, LayerConvertorSm)

    def _forward(self, inputs):
        return self.unetpp(inputs)
