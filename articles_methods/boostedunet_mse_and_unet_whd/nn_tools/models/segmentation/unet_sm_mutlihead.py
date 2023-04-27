import copy
import torch, torch.nn as nn
import segmentation_models_pytorch as sm

from nn_tools.models.layer_convertors import convert_inplace, LayerConvertorSm
from nn_tools.models.segmentation.interface import MultiheadSegmentationNet

class MultiheadUnetSm(MultiheadSegmentationNet):
    def __init__(self, in_channels=3, out_channels_heads=(1,), **kwargs):
        super().__init__(in_channels=in_channels, out_channels_heads=out_channels_heads)

        if len(out_channels_heads) == 0:
            raise ValueError('out_channels_heads is empty')

        bone = sm.Unet(in_channels=in_channels, classes=out_channels_heads[0], **kwargs)

        self.encoder = bone.encoder
        self.decoder = bone.decoder
        convert_inplace(self.decoder, LayerConvertorSm)

        self.__n_heads = len(out_channels_heads)
        self.__suffix = 'head_{}'
        self.heads = nn.ModuleDict() 
        
        self.heads[self.__suffix.format(0)] = bone.segmentation_head

        for idx, out_channels in enumerate(out_channels_heads[1: ]):
            name = self.__suffix.format(idx+1)
            self.heads[name] = copy.deepcopy(bone.segmentation_head)

            old_layer = self.heads[name][0]

            self.heads[name][0] = nn.Conv2d( in_channels=old_layer.in_channels,
                                             out_channels=out_channels,
                                             kernel_size=old_layer.kernel_size,
                                             stride=old_layer.stride,
                                             padding=old_layer.padding,
                                             dilation=old_layer.dilation,
                                             groups=old_layer.groups,
                                             bias='bias' in old_layer.state_dict(),
                                             padding_mode=old_layer.padding_mode )

    def _forward(self, inputs):
        features = self.encoder(inputs)
        decoder_output = self.decoder(*features)

        output = tuple()

        for idx in range(self.__n_heads):
            name = self.__suffix.format(idx)
            output = (*output, self.heads[name](decoder_output))

        return output
