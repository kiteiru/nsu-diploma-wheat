import torch, torch.nn as nn

from nn_tools.models.segmentation.interface import SegmentationNet
from nn_tools.models.segmentation.unet_sm import UnetSm, UnetppSm

def BoostedNet(Type):
    class BoostedNet(SegmentationNet):
        def __init__(self, ntree=3, in_channels=3, out_channels=1, **kwargs):
            super().__init__(in_channels=in_channels, out_channels=out_channels)

            self.__suffix = 'net_{}'
            self.__ntree = ntree

            self.nets = nn.ModuleDict()

            self.nets[self.__suffix.format(0)] = Type(in_channels=in_channels, out_channels=out_channels, **kwargs)

            for idx in range(1, self.__ntree):
                name = self.__suffix.format(len(self.nets))
                self.nets[name] = Type(in_channels=out_channels, out_channels=out_channels, **kwargs)

        def _forward(self, inputs):
            output = (inputs, )

            for idx in range(self.__ntree):
                name = self.__suffix.format(idx)
                output = (*output, self.nets[name](output[-1]))

            if self.training:
                return output[1:]
            else:
                return output[-1]

    return BoostedNet

BoostedUnetSm = BoostedNet(UnetSm)
BoostedUnetppSm = BoostedNet(UnetppSm)
