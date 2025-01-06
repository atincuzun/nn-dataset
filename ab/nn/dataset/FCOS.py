import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Optional, Tuple
from functools import partial
from collections import OrderedDict
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss
from torchvision.ops.boxes import batched_nms

args = [91]

class FCOSHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_convs=4):
        super().__init__()
        self.box_coder = BoxLinearCoder(normalize_by_size=True)
        self.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes, num_convs)
        self.regression_head = FCOSRegressionHead(in_channels, num_anchors, num_convs)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            if len(targets_per_image["labels"]) == 0:
                gt_classes_targets = targets_per_image["labels"].new_zeros((len(matched_idxs_per_image),))
                gt_boxes_targets = targets_per_image["boxes"].new_zeros((len(matched_idxs_per_image), 4))
            else:
                gt_classes_targets = targets_per_image["labels"][matched_idxs_per_image.clip(min=0)]
                gt_boxes_targets = targets_per_image["boxes"][matched_idxs_per_image.clip(min=0)]
            gt_classes_targets[matched_idxs_per_image < 0] = -1  # background
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_boxes_targets)

        all_gt_boxes_targets = torch.stack(all_gt_boxes_targets)
        all_gt_classes_targets = torch.stack(all_gt_classes_targets)
        anchors = torch.stack(anchors)

        # compute foregroud
        foregroud_mask = all_gt_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()

        # classification loss
        cls_logits = head_outputs["cls_logits"]
        gt_classes_targets = torch.zeros_like(cls_logits)
        gt_classes_targets[foregroud_mask, all_gt_classes_targets[foregroud_mask]] = 1.0
        loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="sum")

        # regression loss
        bbox_regression = head_outputs["bbox_regression"]
        pred_boxes = self.box_coder.decode(bbox_regression, anchors)
        loss_bbox_reg = generalized_box_iou_loss(
            pred_boxes[foregroud_mask],
            all_gt_boxes_targets[foregroud_mask],
            reduction="sum",
        )

        # ctrness loss
        bbox_ctrness = head_outputs["bbox_ctrness"]
        bbox_reg_targets = self.box_coder.encode(anchors, all_gt_boxes_targets)
        if len(bbox_reg_targets) == 0:
            gt_ctrness_targets = bbox_reg_targets.new_zeros(bbox_reg_targets.size()[:-1])
        else:
            left_right = bbox_reg_targets[:, :, [0, 2]]
            top_bottom = bbox_reg_targets[:, :, [1, 3]]
            gt_ctrness_targets = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            )
        pred_centerness = bbox_ctrness.squeeze(dim=2)
        loss_bbox_ctrness = nn.functional.binary_cross_entropy_with_logits(
            pred_centerness[foregroud_mask], gt_ctrness_targets[foregroud_mask], reduction="sum"
        )

        return {
            "classification": loss_cls / max(1, num_foreground),
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "bbox_ctrness": loss_bbox_ctrness / max(1, num_foreground),
        }

    def forward(self, x):
        cls_logits = self.classification_head(x)
        bbox_regression, bbox_ctrness = self.regression_head(x)
        return {
            "cls_logits": cls_logits,
            "bbox_regression": bbox_regression,
            "bbox_ctrness": bbox_ctrness,
        }

class FCOSClassificationHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_convs=4, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        norm_layer = partial(nn.GroupNorm, 32)
        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(norm_layer(in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

    def forward(self, x):
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)

class FCOSRegressionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_convs=4):
        super().__init__()
        norm_layer = partial(nn.GroupNorm, 32)

        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(norm_layer(in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.bbox_ctrness = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1)

        for layer in [self.bbox_reg, self.bbox_ctrness]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.zeros_(layer.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        all_bbox_regression = []
        all_bbox_ctrness = []

        for features in x:
            bbox_feature = self.conv(features)
            bbox_regression = nn.functional.relu(self.bbox_reg(bbox_feature))
            bbox_ctrness = self.bbox_ctrness(bbox_feature)

            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)
            all_bbox_regression.append(bbox_regression)

            bbox_ctrness = bbox_ctrness.view(N, -1, 1, H, W)
            bbox_ctrness = bbox_ctrness.permute(0, 3, 4, 1, 2)
            bbox_ctrness = bbox_ctrness.reshape(N, -1, 1)
            all_bbox_ctrness.append(bbox_ctrness)

        return torch.cat(all_bbox_regression, dim=1), torch.cat(all_bbox_ctrness, dim=1)

class BoxLinearCoder:
    def __init__(self, normalize_by_size=True):
        self.normalize_by_size = normalize_by_size

    def encode(self, reference_boxes, proposals):
        device = reference_boxes.device
        dtype = reference_boxes.dtype
        weights = torch.as_tensor([1.0, 1.0, 1.0, 1.0], device=device, dtype=dtype)

        wx = weights[0]
        wy = weights[1]
        ww = weights[2]
        wh = weights[3]

        proposals_x1 = proposals[:, 0].unsqueeze(1)
        proposals_y1 = proposals[:, 1].unsqueeze(1)
        proposals_x2 = proposals[:, 2].unsqueeze(1)
        proposals_y2 = proposals[:, 3].unsqueeze(1)

        reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
        reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
        reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
        reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

        ex_widths = proposals_x2 - proposals_x1
        ex_heights = proposals_y2 - proposals_y1
        if self.normalize_by_size:
            wx = wx / ex_widths
            wy = wy / ex_heights
            ww = ww / ex_widths
            wh = wh / ex_heights

        targets_dx = wx * (reference_boxes_x1 - proposals_x1)
        targets_dy = wy * (reference_boxes_y1 - proposals_y1)
        targets_dw = ww * (reference_boxes_x2 - proposals_x2)
        targets_dh = wh * (reference_boxes_y2 - proposals_y2)

        targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        device = rel_codes.device
        dtype = rel_codes.dtype
        weights = torch.as_tensor([1.0, 1.0, 1.0, 1.0], device=device, dtype=dtype)

        boxes = boxes.to(dtype)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        if self.normalize_by_size:
            widths = 1.0 / widths
            heights = 1.0 / heights
            rel_codes = rel_codes * torch.stack((widths, heights, widths, heights), dim=1)

        boxes = boxes.unsqueeze(1)  # Add a dimension for vectorization
        pred_boxes = boxes + rel_codes * weights

        return pred_boxes

class Net(nn.Module):
    def __init__(self, num_classes=91):
        super().__init__()
        
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        backbone = _resnet_fpn_extractor(
            backbone,
            trainable_layers=3,
            returned_layers=[2, 3, 4],
            extra_blocks=LastLevelP6P7(256, 256)
        )

        # FCOS uses one anchor per location
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios = ((1.0,),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.head = FCOSHead(
            backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            num_classes
        )

        self.box_coder = BoxLinearCoder(normalize_by_size=True)
        
        self.transform = GeneralizedRCNNTransform(
            min_size=320,
            max_size=320,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )

        self.score_thresh = 0.2
        self.nms_thresh = 0.6
        self.detections_per_img = 100
        self.topk_candidates = 1000
        self.num_classes = num_classes
        self.center_sampling_radius = 1.5

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

    def forward(self, images, targets=None):
        if targets is not None:
            # Convert padded tensor format to list of dicts
            batch_size = targets.shape[0]
            dict_targets = []
            
            for i in range(batch_size):
                # Find valid objects (non-zero boxes)
                valid_mask = targets[i, :, :4].sum(dim=1) != 0
                
                dict_targets.append({
                    'boxes': targets[i, valid_mask, :4],
                    'labels': targets[i, valid_mask, 4].long()
                })
            targets = dict_targets

        # Get original image sizes
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # Transform the input
        images = self.transform(images)[0]

        # Get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())

        # Compute the FCOS heads outputs using the features
        head_outputs = self.head(features)

        # Create the set of anchors
        anchors = self.anchor_generator(images, features)
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]

        losses = {}
        detections = []
        
        if self.training:
            # Compute losses
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            
            matched_idxs = []
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image["boxes"].numel() == 0:
                    matched_idxs.append(
                        torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                    )
                    continue

                gt_boxes = targets_per_image["boxes"]
                gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
                anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2
                anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]

                # Center sampling: anchor point must be close enough to gt center
                pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(
                    dim=2
                ).values < self.center_sampling_radius * anchor_sizes[:, None]

                # Compute pairwise distance
                x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)
                x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)
                pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

                # Anchor point must be inside gt
                pairwise_match &= pairwise_dist.min(dim=2).values > 0

                # Each anchor is only responsible for certain scale range
                lower_bound = anchor_sizes * 4
                lower_bound[: num_anchors_per_level[0]] = 0
                upper_bound = anchor_sizes * 8
                upper_bound[-num_anchors_per_level[-1] :] = float("inf")
                pairwise_dist = pairwise_dist.max(dim=2).values
                pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (pairwise_dist < upper_bound[:, None])

                # Match the GT box with minimum area
                gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])
                min_values, matched_idx = pairwise_match.max(dim=1)
                matched_idx[min_values < 1e-5] = -1  # unmatched anchors are assigned -1

                matched_idxs.append(matched_idx)

            losses = self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)
        else:
            # Recover level sizes
            num_anchors_per_level_hw = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level_hw:
                HW += v
            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level_hw]

            # Split outputs per level
            split_head_outputs = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # Compute detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)

        if not self.training:
            return detections

        return losses

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        box_ctrness = head_outputs["bbox_ctrness"]

        num_images = len(image_shapes)
        detections = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            box_ctrness_per_image = [bc[index] for bc in box_ctrness]
            anchors_per_image = anchors[index]
            image_shape = image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, box_ctrness_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, box_ctrness_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # Remove low scoring boxes
                scores_per_level = torch.sqrt(
                    torch.sigmoid(logits_per_level) * torch.sigmoid(box_ctrness_per_level)
                ).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # Keep only topk scoring predictions
                num_topk = min(topk_idxs.size(0), self.topk_candidates)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # Non-maximum suppression
            keep = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections
