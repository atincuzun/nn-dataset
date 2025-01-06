import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Optional, Tuple
from functools import partial
from collections import OrderedDict
from torchvision.models import mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops

args = [91]

def _normal_init(conv: nn.Module):
    """Initialize conv layers with normal distribution (mean=0.0, std=0.03)"""
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)

def _prediction_block(
    in_channels: int, out_channels: int, kernel_size: int, norm_layer: nn.Module
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

def _extra_block(in_channels: int, out_channels: int, norm_layer: nn.Module) -> nn.Sequential:
    activation = nn.ReLU6
    intermediate_channels = out_channels // 2
    return nn.Sequential(
        Conv2dNormActivation(
            in_channels, 
            intermediate_channels, 
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation
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
            intermediate_channels, 
            out_channels, 
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation
        ),
    )

class SSDScoringHead(nn.Module):
    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def _get_result_from_module_list(self, x: Tensor, idx: int) -> Tensor:
        num_blocks = len(self.module_list)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.module_list):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: List[Tensor]) -> Tensor:
        all_results = []

        for i, features in enumerate(x):
            results = self._get_result_from_module_list(features, i)

            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)

            all_results.append(results)

        return torch.cat(all_results, dim=1)

class SSDLiteHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, norm_layer: nn.Module):
        super().__init__()
        self.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
        self.regression_head = SSDLiteRegressionHead(in_channels, num_anchors, norm_layer)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
        }

class SSDLiteClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, norm_layer: nn.Module):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(_prediction_block(channels, num_classes * anchors, 3, norm_layer))
        _normal_init(cls_logits)
        super().__init__(cls_logits, num_classes)

class SSDLiteRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], norm_layer: nn.Module):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(_prediction_block(channels, 4 * anchors, 3, norm_layer))
        _normal_init(bbox_reg)
        super().__init__(bbox_reg, 4)

class Net(nn.Module):
    def __init__(self, num_classes: int = 91):
        super().__init__()

        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1).features

        anchor_generator = DefaultBoxGenerator(
            [[2, 3] for _ in range(6)],
            min_ratio=0.2,
            max_ratio=0.95,
        )

        out_channels = [960, 512, 256, 256, 256, 128]
        num_anchors = anchor_generator.num_anchors_per_location()
        head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)

        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.head = head

        self.box_coder = det_utils.BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.proposal_matcher = det_utils.SSDMatcher(0.5)

        self.score_thresh = 0.001
        self.nms_thresh = 0.55
        self.detections_per_img = 300
        self.topk_candidates = 300
        self.neg_to_pos_ratio = 3

        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(x)
        features = list(features.values())

        head_outputs = self.head(features)
        
        if not self.training:
            pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)
            head_outputs["cls_logits"] = pred_scores

        return head_outputs

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:
        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]

        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        
        for targets_per_image, bbox_regression_per_image, cls_logits_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, cls_logits, anchors, matched_idxs
        ):
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(F.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum"))

            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[foreground_idxs_per_image, targets_per_image["labels"][foreground_matched_idxs_per_image]] = 1.0
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, reduction="none")

        # Hard Negative Mining
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float("inf")
        _, idx = negative_loss.sort(1, descending=True)
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss.sum() / N,
            "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        }

    def postprocess_detections(
        self, head_outputs: Dict[str, Tensor], anchors: List[Tensor], image_shapes: List[Tuple[int, int]]
    ) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = head_outputs["cls_logits"]

        detections = []
        for boxes, scores, anchors_per_image, image_shape in zip(bbox_regression, pred_scores, anchors, image_shapes):
            boxes = self.box_coder.decode_single(boxes, anchors_per_image)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            
            for label in range(1, scores.shape[-1]):
                score = scores[:, label]
                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                keep = box_ops.remove_small_boxes(box, 1e-2)
                box = box[keep]
                score = score[keep]

                keep = box_ops.batched_nms(box, score, torch.zeros_like(score), self.nms_thresh)
                keep = keep[:self.detections_per_img]

                image_boxes.append(box[keep])
                image_scores.append(score[keep])
                image_labels.append(torch.full_like(score[keep], label))

            if image_boxes:
                image_boxes = torch.cat(image_boxes, dim=0)
                image_scores = torch.cat(image_scores, dim=0)
                image_labels = torch.cat(image_labels, dim=0)
            else:
                image_boxes = torch.empty((0, 4))
                image_scores = torch.empty((0,))
                image_labels = torch.empty((0,))

            detections.append(
                {
                    "boxes": image_boxes,
                    "scores": image_scores,
                    "labels": image_labels,
                }
            )
        return detections
        
    def train_setup(self, device, prm):
        self.device = device
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )
        
    def learn(self, train_data):
        self.model.train()
        for inputs, labels in train_data:
            
            inputs = inputs.to(self.device)
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]
        
            optimizer.zero_grad()
        
            loss_dict = self.forward_pass(inputs, labels)
            loss = sum(loss for loss in loss_dict.values())
                
        
            loss.backward()
            optimizer.step()
