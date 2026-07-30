"""
Pure-PyTorch reimplementation of an mmdetection SSD300 (VGG16 + atrous backbone)
that loads the original mmdet checkpoint directly (matching state_dict keys),
with no mmcv/mmdet/mmengine dependency at runtime.
"""

import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, clip_boxes_to_image


# --------------------------------------------------------------------------
# L2Norm (SSD-specific: normalizes conv4_3 features, then rescales per-channel)
# --------------------------------------------------------------------------
class L2Norm(nn.Module):
    def __init__(self, n_channels, eps=1e-10):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        return (x / norm) * self.weight.view(1, -1, 1, 1)


# --------------------------------------------------------------------------
# Backbone: SSDVGG (VGG16 w/ atrous conv6/conv7, indices match state_dict exactly
# because nn.Sequential auto-numbers every child, params or not)
# --------------------------------------------------------------------------
def build_vgg_features():
    cfg = [
        nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),                       # 0,1
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),                      # 2,3
        nn.MaxPool2d(2, 2, ceil_mode=True),                                          # 4
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),                     # 5,6
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),                    # 7,8
        nn.MaxPool2d(2, 2, ceil_mode=True),                                          # 9
        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),                    # 10,11
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),                    # 12,13
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),                    # 14,15
        nn.MaxPool2d(2, 2, ceil_mode=True),                                          # 16
        nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),                    # 17,18
        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),                    # 19,20
        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),                    # 21,22 <- conv4_3 tap (after 22)
        nn.MaxPool2d(2, 2, ceil_mode=True),                                          # 23
        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),                    # 24,25
        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),                    # 26,27
        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),                    # 28,29
        nn.MaxPool2d(3, stride=1, padding=1),                                        # 30
        nn.Conv2d(512, 1024, 3, padding=6, dilation=6), nn.ReLU(inplace=True),       # 31,32
        nn.Conv2d(1024, 1024, 1), nn.ReLU(inplace=True),                             # 33,34 <- fc7 tap (after 34)
    ]
    return nn.Sequential(*cfg)


class SSDVGG(nn.Module):
    CONV4_3_IDX = 22
    FC7_IDX = 34

    def __init__(self):
        super().__init__()
        self.features = build_vgg_features()

    def forward(self, x):
        conv4_3 = fc7 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.CONV4_3_IDX:
                conv4_3 = x
            elif i == self.FC7_IDX:
                fc7 = x
        return conv4_3, fc7


# --------------------------------------------------------------------------
# Neck: SSDNeck (l2_norm on conv4_3 + 4 extra downsampling conv pairs, no BN)
# --------------------------------------------------------------------------
class ConvModuleNoBN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)  # bias=True (default)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.conv(x))


class SSDNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_norm = L2Norm(512)
        self.extra_layers = nn.ModuleList([
            nn.Sequential(ConvModuleNoBN(1024, 256, 1), ConvModuleNoBN(256, 512, 3, stride=2, padding=1)),
            nn.Sequential(ConvModuleNoBN(512, 128, 1), ConvModuleNoBN(128, 256, 3, stride=2, padding=1)),
            nn.Sequential(ConvModuleNoBN(256, 128, 1), ConvModuleNoBN(128, 256, 3, stride=1, padding=0)),
            nn.Sequential(ConvModuleNoBN(256, 128, 1), ConvModuleNoBN(128, 256, 3, stride=1, padding=0)),
        ])

    def forward(self, conv4_3, fc7):
        feats = [self.l2_norm(conv4_3), fc7]
        x = fc7
        for layer in self.extra_layers:
            x = layer(x)
            feats.append(x)
        return feats  # 6 levels: [512, 1024, 512, 256, 256, 256] channels


# --------------------------------------------------------------------------
# Head: SSDHead (single 3x3 conv per level for cls and reg, no shared bridge)
# --------------------------------------------------------------------------
class SSDHead(nn.Module):
    def __init__(self, in_channels=(512, 1024, 512, 256, 256, 256), num_anchors=(4, 6, 6, 6, 4, 4), num_classes=1):
        super().__init__()
        self.cls_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, na * (num_classes + 1), 3, padding=1))
            for c, na in zip(in_channels, num_anchors)
        ])
        self.reg_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, na * 4, 3, padding=1))
            for c, na in zip(in_channels, num_anchors)
        ])
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, feats):
        cls_scores = [conv(f) for conv, f in zip(self.cls_convs, feats)]
        bbox_preds = [conv(f) for conv, f in zip(self.reg_convs, feats)]
        return cls_scores, bbox_preds


# --------------------------------------------------------------------------
# SSD anchor generator (standard SSD300 formula, Liu et al. min/max sizes)
# --------------------------------------------------------------------------
def generate_ssd_anchors(feat_sizes, input_size=300,
                          min_sizes=(21, 45, 99, 153, 207, 261),
                          max_sizes=(45, 99, 153, 207, 261, 315),
                          ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
                          strides=(8, 16, 32, 64, 100, 300),
                          device="cpu"):
    """Returns a list of (num_cells*num_anchors, 4) xyxy anchor tensors, one per level."""
    all_anchors = []
    for (H, W), min_s, max_s, level_ratios, stride in zip(feat_sizes, min_sizes, max_sizes, ratios, strides):
        base_anchors = [(min_s, min_s), (math.sqrt(min_s * max_s), math.sqrt(min_s * max_s))]
        for r in level_ratios:
            sr = math.sqrt(r)
            base_anchors.append((min_s * sr, min_s / sr))
            base_anchors.append((min_s / sr, min_s * sr))
        base_anchors = torch.tensor(base_anchors, device=device)  # (num_anchors, 2) = (w, h)

        shift_y, shift_x = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        cx = (shift_x.float() + 0.5) * stride
        cy = (shift_y.float() + 0.5) * stride
        centers = torch.stack([cx, cy], dim=-1).reshape(-1, 1, 2)  # (H*W, 1, 2)

        wh = base_anchors.unsqueeze(0)  # (1, num_anchors, 2)
        x1y1 = centers - wh / 2
        x2y2 = centers + wh / 2
        level_anchors = torch.cat([x1y1, x2y2], dim=-1).reshape(-1, 4)  # (H*W*num_anchors, 4)
        all_anchors.append(level_anchors)
    return all_anchors


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
class SSD(nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_size=300,
        min_sizes=(21, 45, 99, 153, 207, 261),
        max_sizes=(45, 99, 153, 207, 261, 315),
        anchor_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
        strides=(8, 16, 32, 64, 100, 300),
        bbox_coder_stds=(0.1, 0.1, 0.2, 0.2),   # mmdet SSD default target_stds
        image_mean=(123.675, 116.28, 103.53),
        image_std=(1.0, 1.0, 1.0),   # mmdet's SSDVGG (caffe-style) uses std=1, bgr order handled by to_rgb flag
        score_thresh=0.02,
        nms_thresh=0.45,
        max_per_img=200,
    ):
        super().__init__()
        self.backbone = SSDVGG()
        self.neck = SSDNeck()
        self.bbox_head = SSDHead(num_classes=num_classes)

        self.register_buffer("image_mean", torch.tensor(image_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std).view(-1, 1, 1), persistent=False)

        self.input_size = input_size
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.anchor_ratios = anchor_ratios
        self.strides = strides
        self.bbox_coder_stds = bbox_coder_stds
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_per_img = max_per_img

    def normalize(self, img):
        return img - self.image_mean  # std=1 for caffe-style VGG; adjust if your config differs

    def _decode_boxes(self, anchors, deltas):
        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        acx = anchors[:, 0] + aw / 2
        acy = anchors[:, 1] + ah / 2

        dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
        sx, sy, sw, sh = self.bbox_coder_stds
        pcx = acx + dx * sx * aw
        pcy = acy + dy * sy * ah
        pw = aw * torch.exp(dw * sw)
        ph = ah * torch.exp(dh * sh)

        return torch.stack([pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2], dim=-1)

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
        scale_factor = min(300/H, 300/W)
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
        device = x.device

        conv4_3, fc7 = self.backbone(x)
        feats = self.neck(conv4_3, fc7)
        cls_scores, bbox_preds = self.bbox_head(feats)

        feat_sizes = [f.shape[-2:] for f in feats]
        anchors_per_level = generate_ssd_anchors(
            feat_sizes, self.input_size, self.min_sizes, self.max_sizes,
            self.anchor_ratios, self.strides, device=device,
        )

        all_boxes, all_scores, all_labels = [], [], []
        for cls_s, bbox_p, anchors, na in zip(
            cls_scores, bbox_preds, anchors_per_level, self.bbox_head.num_anchors
        ):
            H, W = cls_s.shape[-2:]
            scores = cls_s[0].permute(1, 2, 0).reshape(-1, self.num_classes + 1).softmax(dim=-1)
            deltas = bbox_p[0].permute(1, 2, 0).reshape(-1, 4)
            boxes = self._decode_boxes(anchors, deltas)

            for cls_idx in range(self.num_classes):  # last column (index -1) = background, mmdet convention
                s = scores[:, cls_idx]
                keep = s > self.score_thresh
                if keep.sum() == 0:
                    continue
                all_boxes.append(boxes[keep])
                all_scores.append(s[keep])
                all_labels.append(torch.full((keep.sum(),), cls_idx, dtype=torch.long, device=device))

        if not all_boxes:
            return {"bboxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}

        boxes = clip_boxes_to_image(torch.cat(all_boxes), image_shape)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)

        keep = batched_nms(boxes, scores, labels, self.nms_thresh)
        keep = keep[: self.max_per_img]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        boxes = (boxes.detach().cpu().numpy() * scale_factor).tolist()
        scores = scores.detach().cpu().numpy().tolist()
        labels = labels.detach().cpu().numpy().tolist()

        return {"bboxes": boxes, "scores": scores, "labels": labels}
