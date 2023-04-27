import torch, torch.nn as nn
import segmentation_models_pytorch as sm

class UnetSm(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()
        
        self.unet = sm.Unet(in_channels=in_channels, classes=out_channels, **kwargs)

    def forward(self, inputs):
        return self.unet(inputs)

class UnetppSm(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()

        self.unetpp = sm.UnetPlusPlus(in_channels=in_channels, classes=out_channels, **kwargs)

    def forward(self, inputs):
        return self.unetpp(inputs)
