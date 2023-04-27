import torch.nn as nn

from nn_tools.exceptions import ShapeNNToolsException

class SegmentationNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

    def forward(self, x):
        if len(x.shape) != 4:
            raise ShapeNNToolsException

        if x.shape[1] != self._in_channels:
            raise ShapeNNToolsException

        y = self._forward(x)

        if not self.training:
            if y.shape[1] != self._out_channels:
                raise ShapeNNToolsException

        return y

    def _forward(self, x):
        raise NotImplemented(f'Not implemented _forward in {self.__class__}')

class MultiheadSegmentationNet(nn.Module):
    def __init__(self, in_channels=3, out_channels_heads=(1,), **kwargs):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels_heads = out_channels_heads

    def forward(self, x):
        if len(x.shape) != 4:
            raise ShapeNNToolsException

        if x.shape[1] != self._in_channels:
            raise ShapeNNToolsException

        y = self._forward(x)

        if not self.training:
            if len(y) != len(self._out_channels_heads):
                raise ShapeNNToolsException

                for yi, out_channels in zip(y, self._out_channels_heads):
                    if yi.shape[1] != out_channels:
                        raise ShapeNNToolsException

        return y

    def _forward(self, x):
        raise NotImplemented(f'Not implemented _forward in {self.__class__}')
