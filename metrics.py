import torch.nn as nn
import torch
from monai.metrics import MSEMetric, PSNRMetric, MAEMetric, SSIMMetric
from monai.metrics import DiceMetric, SurfaceDiceMetric, HausdorffDistanceMetric


class SegMetrics(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        
        self.dice = DiceMetric(include_background=True,
                               num_classes=num_classes)

        self.nsd = SurfaceDiceMetric(include_background=True,
                               class_thresholds=[4, 4, 2])

        self.hd = HausdorffDistanceMetric(include_background=True,
                               percentile=95)
        
    def forward(self, x, y):
        dice = self.dice(x, y)
        nsd = self.nsd(x, y)
        hd = torch.nan_to_num(self.hd(x, y), 0.)
        
        return dice, nsd, hd


class GenMetrics(nn.Module):
    def __init__(self, spatial_dims) -> None:
        super().__init__()
        
        self.mae = MAEMetric()
        self.mse = MSEMetric()
        self.psnr = PSNRMetric(max_val=50)
        self.ssim = SSIMMetric(spatial_dims)
        
    def forward(self, x, y):
        return self.mae(x, y), self.mse(x, y), self.psnr(x, y), self.ssim(x, y)
   