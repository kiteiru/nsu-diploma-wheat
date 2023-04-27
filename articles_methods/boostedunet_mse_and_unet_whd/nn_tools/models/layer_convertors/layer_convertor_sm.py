import torch
import types
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as sm

from nn_tools.models.layer_convertors.misc import __classinit
from nn_tools.models.layer_convertors.layer_convertor import LayerConvertor


@__classinit
class LayerConvertorSm(LayerConvertor.__class__):
    @classmethod
    def _init__class(cls):
        cls._registry = {
            sm.unet.decoder.DecoderBlock: getattr(cls, '_func_sm_unet_DecoderBlock'),
            sm.unetplusplus.decoder.DecoderBlock: getattr(cls, '_func_sm_unetplusplus_DecoderBlock')
        }

        return cls()

    @staticmethod
    def _sm_unet_decoder_forward(self, x, skip=None):
        if skip is not None:
            scale_factor = list()

            getdim = lambda vector, axis : vector.shape[axis]

            naxis = len(x.shape)
            for axis in np.arange(2, naxis):
                scale_factor.append(getdim(skip, axis)/getdim(x, axis))

            scale_factor = tuple(scale_factor)
        else:
            scale_factor = 2

        x = nn.functional.interpolate(x, scale_factor=scale_factor, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

    @classmethod
    def _func_sm_unet_DecoderBlock(cls, layer):
        layer.forward = types.MethodType(cls._sm_unet_decoder_forward, layer)

        return layer

    @classmethod
    def _func_sm_unetplusplus_DecoderBlock(cls, layer):
        layer.forward = types.MethodType(cls._sm_unet_decoder_forward, layer)

        return layer
