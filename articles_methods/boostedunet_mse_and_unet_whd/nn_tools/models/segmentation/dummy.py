import torch

from nn_tools.models.segmentation.interface import SegmentationNet

class DummySegmentationNet(SegmentationNet):
    def _forward(self, x):
        return torch.randn((x.shape[0], self._out_channels, *x.shape[1:]), requires_grad=True)
