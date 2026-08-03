"""
Pure-PyTorch reimplementation of an mmdetection RTMDet (CSPNeXt backbone +
CSPNeXtPAFPN neck + anchor-free RTMDetSepBNHead) that loads the original mmdet
checkpoint directly (matching state_dict keys), with no mmcv/mmdet/mmengine
dependency at runtime.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, clip_boxes_to_image


# --------------------------------------------------------------------------
# Basic building blocks
# --------------------------------------------------------------------------
class ConvModule(nn.Module):
    """conv(no bias) + bn + SiLU, matching mmcv's default CSPNeXt config."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, channels, kernel_size=5, padding=2):
        super().__init__()
        self.depthwise_conv = ConvModule(channels, channels, kernel_size, padding=padding, groups=channels)
        self.pointwise_conv = ConvModule(channels, channels, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        return x * self.act(self.fc(self.global_avgpool(x)))


class CSPNeXtBlock(nn.Module):
    """3x3 conv -> depthwise-separable 5x5, same channels throughout, residual add."""
    def __init__(self, channels,
                 add_identity=True, use_depthwise=False):
        super().__init__()
        # self.conv1 = ConvModule(channels, channels, 3, padding=1)
        self.conv1 = DepthwiseSeparableConvModule(channels, channels, 3, padding=1) if use_depthwise else ConvModule(channels, channels, 3, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(channels)
        # self.conv2 = DepthwiseSeparableConvModule(channels) if use_depthwise else ConvModule(channels)
        self.add_identity = add_identity

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.add_identity:
            return out + x
        else:
            return out


class CSPLayer(nn.Module):
    """Dual-path (main/short) CSP block with N internal CSPNeXtBlocks, optional
    ChannelAttention at the end (present in backbone stages, absent in the neck)."""
    def __init__(self, in_ch, out_ch, num_blocks, use_attention=False,
                 add_identity=True, use_depthwise=False):
        super().__init__()
        mid_ch = out_ch // 2
        self.main_conv = ConvModule(in_ch, mid_ch, 1)
        self.short_conv = ConvModule(in_ch, mid_ch, 1)
        self.final_conv = ConvModule(mid_ch * 2, out_ch, 1)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(mid_ch, add_identity=add_identity, use_depthwise=use_depthwise) for _ in range(num_blocks)])
        self.attention = ChannelAttention(out_ch) if use_attention else None

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        x_final = torch.cat([x_main, x_short], dim=1)
        if self.attention is not None:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class SPPBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, pool_sizes=(5, 9, 13)):
        super().__init__()
        mid_ch = in_ch // 2
        self.conv1 = ConvModule(in_ch, mid_ch, 1)
        self.poolings = nn.ModuleList(
            [nn.MaxPool2d(k, stride=1, padding=k // 2) for k in pool_sizes]
        )
        self.conv2 = ConvModule(mid_ch * (len(pool_sizes) + 1), out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [p(x) for p in self.poolings], dim=1)
        return self.conv2(x)


# --------------------------------------------------------------------------
# Backbone: CSPNeXt
# --------------------------------------------------------------------------
class CSPNeXt(nn.Module):
    # (out_channels, num_csp_blocks, use_spp, use_attention) per stage
    stage_cfg = [
        (128, 3, False, True, True),   # stage1
        (256, 6, False, True, True),   # stage2
        (512, 6, False, True, True),   # stage3
        (1024, 3, True, True, False),   # stage4
    ]

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvModule(3, 32, 3, stride=2, padding=1),
            ConvModule(32, 32, 3, stride=1, padding=1),
            ConvModule(32, 64, 3, stride=1, padding=1),
        )
        in_ch = 64
        for i, (out_ch, n_blocks, use_spp, use_attn, add_identity) in enumerate(self.stage_cfg, start=1):
            layers = [ConvModule(in_ch, out_ch, 3, stride=2, padding=1)]
            if use_spp:
                layers.append(SPPBottleneck(out_ch, out_ch))
            layers.append(CSPLayer(out_ch, out_ch, n_blocks, use_attention=use_attn,
                                   add_identity=add_identity,
                                   use_depthwise=False))
            setattr(self, f"stage{i}", nn.Sequential(*layers))
            in_ch = out_ch

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)   # stride 8,  256 ch
        c4 = self.stage3(c3)  # stride 16, 512 ch
        c5 = self.stage4(c4)  # stride 32, 1024 ch
        return c3, c4, c5


# --------------------------------------------------------------------------
# Neck: CSPNeXtPAFPN
# --------------------------------------------------------------------------
class CSPNeXtPAFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.reduce_layers = nn.ModuleList([
            ConvModule(1024, 512, 1),
            ConvModule(512, 256, 1),
        ])
        self.top_down_blocks = nn.ModuleList([
            CSPLayer(1024, 512, num_blocks=3, use_attention=False, add_identity=False),
            CSPLayer(512, 256, num_blocks=3, use_attention=False, add_identity=False),
        ])
        self.downsamples = nn.ModuleList([
            ConvModule(256, 256, 3, stride=2, padding=1),
            ConvModule(512, 512, 3, stride=2, padding=1),
        ])
        self.bottom_up_blocks = nn.ModuleList([
            CSPLayer(512, 512, num_blocks=3, use_attention=False, add_identity=False),
            CSPLayer(1024, 1024, num_blocks=3, use_attention=False, add_identity=False),
        ])
        self.out_convs = nn.ModuleList([
            ConvModule(256, 256, 3, padding=1),
            ConvModule(512, 256, 3, padding=1),
            ConvModule(1024, 256, 3, padding=1),
        ])

    def forward(self, inputs):
        # Code adapted from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/cspnext_pafpn.py
        num_inputs = 3
        assert len(inputs) == num_inputs

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(num_inputs - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[num_inputs - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[num_inputs - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(num_inputs - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)


# --------------------------------------------------------------------------
# Head: RTMDetSepBNHead (anchor-free, per-level independent conv stacks)
# --------------------------------------------------------------------------
class RTMDetSepBNHead(nn.Module):
    def __init__(self, num_levels=3, in_channels=256, feat_channels=256, stacked_convs=2, num_classes=1):
        super().__init__()
        self.cls_convs = nn.ModuleList([
            nn.ModuleList([ConvModule(in_channels if j == 0 else feat_channels, feat_channels, 3, padding=1)
                           for j in range(stacked_convs)])
            for _ in range(num_levels)
        ])
        self.reg_convs = nn.ModuleList([
            nn.ModuleList([ConvModule(in_channels if j == 0 else feat_channels, feat_channels, 3, padding=1)
                           for j in range(stacked_convs)])
            for _ in range(num_levels)
        ])
        self.rtm_cls = nn.ModuleList([nn.Conv2d(feat_channels, num_classes, 1) for _ in range(num_levels)])
        self.rtm_reg = nn.ModuleList([nn.Conv2d(feat_channels, 4, 1) for _ in range(num_levels)])
        self.num_classes = num_classes

    def forward(self, feats):
        strides = [(8, 8),
                   (16, 16),
                   (32, 32),]
        assert len(strides) == len(feats)
        cls_scores, bbox_preds = [], []
        for i, (feat, stride) in enumerate(zip(feats, strides)):
            cls_feat = feat
            for conv in self.cls_convs[i]:
                cls_feat = conv(cls_feat)
            cls_scores.append(self.rtm_cls[i](cls_feat))

            reg_feat = feat
            for conv in self.reg_convs[i]:
                reg_feat = conv(reg_feat)
            bbox_preds.append(self.rtm_reg[i](reg_feat))
        return cls_scores, bbox_preds


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
class RTMDet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        strides=(8, 16, 32),
        image_mean=(103.53, 116.28, 123.675),
        image_std=(57.375, 57.12, 58.395),
        score_thresh=0.0,
        nms_thresh=0.65,
        max_per_img=300,
    ):
        super().__init__()
        self.backbone = CSPNeXt()
        self.neck = CSPNeXtPAFPN()
        self.bbox_head = RTMDetSepBNHead(num_classes=num_classes)

        self.register_buffer("image_mean", torch.tensor(image_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std).view(-1, 1, 1), persistent=False)

        self.strides = strides
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_per_img = max_per_img

    def normalize(self, img):
        return (img - self.image_mean) / self.image_std

    @staticmethod
    def _get_points(H, W, stride, device):
        shift_y, shift_x = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        cx = shift_x.float() * stride
        cy = shift_y.float() * stride
        return torch.stack([cx, cy], dim=-1).reshape(-1, 2)  # (H*W, 2)

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
        scale_factor = min(640/H, 640/W)
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

        c3, c4, c5 = self.backbone(x)
        feats = self.neck((c3, c4, c5))
        cls_scores, bbox_preds = self.bbox_head(feats)

        all_boxes, all_scores, all_labels = [], [], []
        for cls_s, bbox_p, stride in zip(cls_scores, bbox_preds, self.strides):
            H, W = cls_s.shape[-2:]
            points = self._get_points(H, W, stride, device)  # (H*W, 2)

            scores = cls_s[0].permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
            dists = bbox_p[0].permute(1, 2, 0).reshape(-1, 4).exp() * stride  # (l, t, r, b) in pixels

            x1 = points[:, 0] - dists[:, 0]  # -
            y1 = points[:, 1] - dists[:, 1]  # -
            x2 = points[:, 0] + dists[:, 2]
            y2 = points[:, 1] + dists[:, 3]
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            for cls_idx in range(self.num_classes):
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
