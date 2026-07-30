"""
Pure-PyTorch reimplementation of an mmdetection Faster R-CNN (ResNet50-FPN)
that loads the original mmdet checkpoint directly (matching state_dict keys),
with no mmcv/mmdet/mmengine dependency at runtime.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.ops import RoIAlign, batched_nms, clip_boxes_to_image, remove_small_boxes
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection._utils import BoxCoder


# --------------------------------------------------------------------------
# Backbone (ResNet50, torchvision-native naming)
# --------------------------------------------------------------------------
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        r = resnet50(weights=None)
        self.conv1 = r.conv1
        self.bn1 = r.bn1
        self.relu = r.relu
        self.maxpool = r.maxpool
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.layer4 = r.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]  # strides 4, 8, 16, 32


# --------------------------------------------------------------------------
# FPN neck (mmdet naming: lateral_convs / fpn_convs, plain conv, has bias)
# --------------------------------------------------------------------------
class FPN(nn.Module):
    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [_ConvModule(c, out_channels, kernel_size=1) for c in in_channels]
        )
        self.fpn_convs = nn.ModuleList(
            [_ConvModule(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels]
        )
        self.extra_maxpool = nn.MaxPool2d(kernel_size=1, stride=2)

    def forward(self, feats):
        laterals = [lc(f) for lc, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        outs = [fc(l) for fc, l in zip(self.fpn_convs, laterals)]
        outs.append(self.extra_maxpool(outs[-1]))  # P6, no learnable params
        return outs  # 5 levels: P2..P6 (strides 4,8,16,32,64)


class _ConvModule(nn.Module):
    """Matches mmcv ConvModule's `.conv` attribute path with no norm/act,
    so state_dict keys read e.g. 'neck.lateral_convs.0.conv.weight'."""
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)


# --------------------------------------------------------------------------
# RPN head (mmdet naming: rpn_conv / rpn_cls / rpn_reg, weight-shared across levels)
# --------------------------------------------------------------------------
class RPNHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=3):
        super().__init__()
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(in_channels, num_anchors, 1)       # sigmoid objectness per anchor
        self.rpn_reg = nn.Conv2d(in_channels, num_anchors * 4, 1)   # box deltas per anchor
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats):
        cls_scores, bbox_preds = [], []
        for f in feats:
            t = self.relu(self.rpn_conv(f))
            cls_scores.append(self.rpn_cls(t))
            bbox_preds.append(self.rpn_reg(t))
        return cls_scores, bbox_preds


# --------------------------------------------------------------------------
# ROI extractor (RoIAlign per FPN level; no learnable params -> no state_dict keys)
# --------------------------------------------------------------------------
class SingleRoIExtractor(nn.Module):
    def __init__(self, output_size=7, featmap_strides=(4, 8, 16, 32), finest_scale=56):
        super().__init__()
        self.output_size = output_size
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.roi_layers = nn.ModuleList(
            [RoIAlign((output_size, output_size), spatial_scale=1.0 / s,
                      sampling_ratio=0, aligned=True) for s in featmap_strides]
        )

    def map_roi_levels(self, rois, num_levels):
        # mmdet's formula: level = floor(log2(sqrt(area) / finest_scale + eps))
        scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        return target_lvls.clamp(min=0, max=num_levels - 1).long()

    def forward(self, feats, rois):
        """rois: (N, 5) tensor of [batch_idx, x1, y1, x2, y2]"""
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        out = rois.new_zeros(rois.size(0), feats[0].size(1), self.output_size, self.output_size)
        for lvl in range(num_levels):
            mask = target_lvls == lvl
            if mask.any():
                out[mask] = self.roi_layers[lvl](feats[lvl], rois[mask])
        return out


# --------------------------------------------------------------------------
# Box head (mmdet naming: shared_fcs.0/1, fc_cls, fc_reg -- class-agnostic reg)
# --------------------------------------------------------------------------
class Shared2FCBBoxHead(nn.Module):
    def __init__(self, roi_feat_size=7, in_channels=256, fc_out=1024, num_classes=1):
        super().__init__()
        self.shared_fcs = nn.ModuleList(
            [nn.Linear(in_channels * roi_feat_size * roi_feat_size, fc_out),
             nn.Linear(fc_out, fc_out)]
        )
        self.fc_cls = nn.Linear(fc_out, num_classes + 1)  # +1 background
        self.fc_reg = nn.Linear(fc_out, 4)                # class-agnostic
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.flatten(1)
        for fc in self.shared_fcs:
            x = self.relu(fc(x))
        return self.fc_cls(x), self.fc_reg(x)


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
class FasterRCNN(nn.Module):
    def __init__(
        self,
        num_classes=1,               # foreground classes only (background added automatically)
        image_mean=(123.675, 116.28, 103.53),
        image_std=(58.395, 57.12, 57.375),
        anchor_scales=(8,),
        anchor_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(4, 8, 16, 32, 64),
        rpn_pre_nms_top_n=1000,
        rpn_post_nms_top_n=1000,
        rpn_nms_thresh=0.7,
        rpn_score_thresh=0.0,
        bbox_coder_weights=(1.0, 1.0, 1.0, 1.0),
        roi_bbox_coder_weights=(10.0, 10.0, 5.0, 5.0),
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    ):
        super().__init__()
        self.backbone = Backbone()
        self.neck = FPN()
        self.rpn_head = RPNHead(num_anchors=len(anchor_ratios) * len(anchor_scales))

        self.roi_head = nn.Module()
        self.roi_head.bbox_roi_extractor = SingleRoIExtractor(featmap_strides=anchor_strides[:4])
        self.roi_head.bbox_head = Shared2FCBBoxHead(num_classes=num_classes)

        # --- inference-only config (not learned, not in state_dict) ---
        self.register_buffer("image_mean", torch.tensor(image_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std).view(-1, 1, 1), persistent=False)

        sizes = tuple((s,) for s in anchor_scales) * len(anchor_strides) if False else \
            tuple((s * st,) for st in anchor_strides for s in anchor_scales)
        # AnchorGenerator expects one size-tuple and one ratio-tuple PER feature level
        self.anchor_generator = AnchorGenerator(
            sizes=tuple((s * st,) for st in anchor_strides for s in anchor_scales[:1]),
            aspect_ratios=(anchor_ratios,) * len(anchor_strides),
        )
        self.rpn_box_coder = BoxCoder(weights=bbox_coder_weights)
        self.roi_box_coder = BoxCoder(weights=roi_bbox_coder_weights)

        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.rpn_nms_thresh = rpn_nms_thresh
        self.rpn_score_thresh = rpn_score_thresh
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.box_detections_per_img = box_detections_per_img
        self.num_classes = num_classes

    # ---------------- preprocessing ----------------
    def normalize(self, img):
        # return img
        return (img - self.image_mean) / self.image_std

    # ---------------- RPN proposal decoding ----------------
    def _get_proposals(self, feats, anchors_per_level, cls_scores, bbox_preds, image_shape):
        device = feats[0].device
        all_boxes, all_scores = [], []
        for anchors, cls_s, bbox_p in zip(anchors_per_level, cls_scores, bbox_preds):
            N, A, H, W = bbox_p.shape[0], cls_s.shape[1], cls_s.shape[2], cls_s.shape[3]
            scores = cls_s.permute(0, 2, 3, 1).reshape(N, -1)  # (N, H*W*A)
            deltas = bbox_p.permute(0, 2, 3, 1).reshape(N, -1, 4)

            scores = torch.sigmoid(scores)
            top_n = min(self.rpn_pre_nms_top_n, scores.shape[1])
            top_scores, top_idx = scores.topk(top_n, dim=1)
            boxes = self.rpn_box_coder.decode(
                deltas[0, top_idx[0]], [anchors[top_idx[0]]]
            ).reshape(-1, 4)
            boxes = clip_boxes_to_image(boxes, image_shape)
            all_boxes.append(boxes)
            all_scores.append(top_scores[0])

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        keep = remove_small_boxes(boxes, min_size=1e-3)
        boxes, scores = boxes[keep], scores[keep]
        keep = scores >= self.rpn_score_thresh
        boxes, scores = boxes[keep], scores[keep]
        keep = batched_nms(boxes, scores, torch.zeros_like(scores, dtype=torch.long), self.rpn_nms_thresh)
        keep = keep[: self.rpn_post_nms_top_n]
        return boxes[keep]

    # ---------------- final box decoding ----------------
    def _postprocess_boxes(self, cls_logits, bbox_deltas, proposals, image_shape):
        scores = F.softmax(cls_logits, dim=-1)
        num_classes = scores.shape[1]  # includes background as last or first index (mmdet: last)
        boxes = self.roi_box_coder.decode(bbox_deltas, [proposals]).reshape(-1, 4)
        boxes = clip_boxes_to_image(boxes, image_shape)

        all_boxes, all_scores, all_labels = [], [], []
        for cls_idx in range(num_classes - 1):  # skip background (last column, mmdet convention)
            cls_scores = scores[:, cls_idx]
            keep = cls_scores > self.box_score_thresh
            if keep.sum() == 0:
                continue
            b, s = boxes[keep], cls_scores[keep]
            nms_keep = batched_nms(b, s, torch.full_like(s, cls_idx, dtype=torch.long), self.box_nms_thresh)
            all_boxes.append(b[nms_keep])
            all_scores.append(s[nms_keep])
            all_labels.append(torch.full((len(nms_keep),), cls_idx, dtype=torch.long, device=b.device))

        if not all_boxes:
            return (torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long))

        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        if len(scores) > self.box_detections_per_img:
            top = scores.topk(self.box_detections_per_img).indices
            boxes, scores, labels = boxes[top], scores[top], labels[top]
        return boxes, scores, labels

    def preprocess(self, image):
        assert isinstance(image, np.ndarray), 'Input should be a numpy array of shape (H, W, C)'
        
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        assert image.ndim == 3, f'Input should be a numpy array of shape (H, W, C), not {image.shape}'

        if image.shape[2]==1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2]==3:
            pass
        elif image.shape[2]==4:
            # Remove alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            raise RuntimeError(f'Input should be a numpy array of shape (H, W, C), with C either 1 (grayscale), 3 (RGB), or 4 (RGBA), not {image.shape}.')

        # Resize image
        H, W = image.shape[:2]
        scale_factor = min(1333/H, 800/W)
        H_new, W_new = int(H*scale_factor), int(W*scale_factor)
        image = cv2.resize(image, (H_new, W_new), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float)
        
        return image, 1 / scale_factor

    # ---------------- forward (single image, inference only) ----------------
    @torch.no_grad()
    def forward(self, image):
        """image: (H, W, 3) uint8 tensor."""
        image, scale_factor = self.preprocess(image)
        image_shape = image.shape[-2:]
        x = self.normalize(image).unsqueeze(0)

        feats = self.backbone(x)
        feats = self.neck(feats)

        cls_scores, bbox_preds = self.rpn_head(feats)
        anchors_per_level = self.anchor_generator(
            _FakeImageList(x), feats
        )
        # AnchorGenerator returns anchors per-image; we only have 1 image, but they're
        # already concatenated across levels internally by torchvision -- split them
        # back out by level using each feature map's cell count.
        anchors_per_level = self._split_anchors_by_level(anchors_per_level[0], feats)

        proposals = self._get_proposals(feats, anchors_per_level, cls_scores, bbox_preds, image_shape)
        rois = torch.cat(
            [torch.zeros(proposals.shape[0], 1, device=proposals.device), proposals], dim=1
        )
        roi_feats = self.roi_head.bbox_roi_extractor(feats[:4], rois)
        cls_logits, bbox_deltas = self.roi_head.bbox_head(roi_feats)

        # expand deltas per-class if needed (here class-agnostic, so just reuse)
        boxes, scores, labels = self._postprocess_boxes(cls_logits, bbox_deltas, proposals, image_shape)
        
        if boxes.shape[0]==0:
            return {"bboxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}
        
        boxes = (boxes.detach().cpu().numpy() * scale_factor).tolist()
        scores = scores.detach().cpu().numpy().tolist()
        labels = labels.detach().cpu().numpy().tolist()
        return {"bboxes": boxes, "scores": scores, "labels": labels}

    @staticmethod
    def _split_anchors_by_level(anchors, feats):
        out, i = [], 0
        for f in feats:
            n = f.shape[-2] * f.shape[-1] * (anchors.shape[0] // sum(ff.shape[-2] * ff.shape[-1] for ff in feats))
            out.append(anchors[i : i + n])
            i += n
        return out


class _FakeImageList:
    """Minimal stand-in for torchvision's ImageList, since AnchorGenerator only
    reads .image_sizes and .tensors.shape[-2:] from it."""
    def __init__(self, tensor):
        self.tensors = tensor
        self.image_sizes = [tuple(tensor.shape[-2:])]

