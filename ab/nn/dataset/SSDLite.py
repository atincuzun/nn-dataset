import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Callable
from functools import partial
from torchvision.models.detection.ssd import SSD, SSDScoringHead
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models import mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

args = [91]  

def _prediction_block(
    in_channels: int, out_channels: int, kernel_size: int, norm_layer: Callable[..., nn.Module]
) -> nn.Sequential:
    return nn.Sequential(
        Conv2dNormActivation(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU6,
        ),
        nn.Conv2d(in_channels, out_channels, 1),
    )

def _extra_block(in_channels: int, out_channels: int, norm_layer: Callable[..., nn.Module]) -> nn.Sequential:
    activation = nn.ReLU6
    intermediate_channels = out_channels // 2
    return nn.Sequential(
        Conv2dNormActivation(
            in_channels, intermediate_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=activation
        ),
        Conv2dNormActivation(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=2,
            groups=intermediate_channels,
            norm_layer=norm_layer,
            activation_layer=activation,
        ),
        Conv2dNormActivation(
            intermediate_channels, out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=activation
        ),
    )

class SSDLiteHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, norm_layer: Callable[..., nn.Module]):
        super().__init__()
        self.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
        self.regression_head = SSDLiteRegressionHead(in_channels, num_anchors, norm_layer)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
        }

class SSDLiteClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, norm_layer: Callable[..., nn.Module]):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(_prediction_block(channels, num_classes * anchors, 3, norm_layer))
        super().__init__(cls_logits, num_classes)

class SSDLiteRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], norm_layer: Callable[..., nn.Module]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(_prediction_block(channels, 4 * anchors, 3, norm_layer))
        super().__init__(bbox_reg, 4)

class Net(SSD):
    def __init__(self, num_classes: int = 91):
        # Get backbone from torchvision
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1).features
        
        # Create anchor generator
        anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
        
        # Setup norm layer
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        
        # Create head
        out_channels = [960, 512, 256, 256, 256, 128]
        num_anchors = anchor_generator.num_anchors_per_location()
        head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)
        
        # Initialize SSD parent class
        super().__init__(
            backbone=backbone,
            anchor_generator=anchor_generator,
            size=(320, 320),  # SSDLite uses 320x320
            num_classes=num_classes,
            head=head,
            score_thresh=0.001,
            nms_thresh=0.55,
            detections_per_img=300,
            image_mean=[0.5, 0.5, 0.5],  # As per SSDLite defaults
            image_std=[0.5, 0.5, 0.5]
        )
    

