"""
Pure-PyTorch reimplementation of an mmdetection YOLOv3 (Darknet-53 backbone)
that loads the original mmdet checkpoint directly (matching state_dict keys),
with no mmcv/mmdet/mmengine dependency at runtime.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, clip_boxes_to_image


# --------------------------------------------------------------------------
# Basic building block (mmcv ConvModule: conv(no bias) + bn + LeakyReLU(0.1))
# --------------------------------------------------------------------------
class ConvModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activate = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """1x1 channel-reduce -> 3x3 restore, with residual add (Darknet-53 style)."""
    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.conv1 = ConvModule(channels, half, kernel_size=1)
        self.conv2 = ConvModule(half, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


def make_res_block_layer(in_ch, out_ch, num_blocks):
    """Matches mmdet's `conv_res_blockN`: downsampling conv + N ResBlocks."""
    layer = nn.Sequential()
    layer.add_module("conv", ConvModule(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
    for i in range(num_blocks):
        layer.add_module(f"res{i}", ResBlock(out_ch))
    return layer


# --------------------------------------------------------------------------
# Backbone: Darknet-53
# --------------------------------------------------------------------------
class Darknet53(nn.Module):
    # (out_channels, num_res_blocks) per stage -- standard Darknet-53 config
    layer_cfg = [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)]

    def __init__(self):
        super().__init__()
        self.conv1 = ConvModule(3, 32, kernel_size=3, padding=1)
        in_ch = 32
        for i, (out_ch, n) in enumerate(self.layer_cfg, start=1):
            setattr(self, f"conv_res_block{i}", make_res_block_layer(in_ch, out_ch, n))
            in_ch = out_ch

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_res_block1(x)
        x = self.conv_res_block2(x)
        c3 = self.conv_res_block3(x)   # stride 8,  256 ch
        c4 = self.conv_res_block4(c3)  # stride 16, 512 ch
        c5 = self.conv_res_block5(c4)  # stride 32, 1024 ch
        return c3, c4, c5


# --------------------------------------------------------------------------
# Neck: YOLOV3Neck (5-conv DetectionBlock per scale + top-down upsample/concat)
# --------------------------------------------------------------------------
class DetectionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        half = out_ch // 2
        self.conv1 = ConvModule(in_ch, half, kernel_size=1)
        self.conv2 = ConvModule(half, out_ch, kernel_size=3, padding=1)
        self.conv3 = ConvModule(out_ch, half, kernel_size=1)
        self.conv4 = ConvModule(half, out_ch, kernel_size=3, padding=1)
        self.conv5 = ConvModule(out_ch, half, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)  # "route" feature, half of out_ch


class YOLOV3Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.detect1 = DetectionBlock(1024, 1024)   # -> 512 ch
        self.conv1 = ConvModule(512, 256, kernel_size=1)
        self.detect2 = DetectionBlock(768, 512)      # 256(up) + 512(c4) = 768 -> 256 ch
        self.conv2 = ConvModule(256, 128, kernel_size=1)
        self.detect3 = DetectionBlock(384, 256)      # 128(up) + 256(c3) = 384 -> 128 ch

    def forward(self, c3, c4, c5):
        out1 = self.detect1(c5)                                   # 512 ch, stride 32
        x = self.conv1(out1)
        x = F.interpolate(x, size=c4.shape[-2:], mode="nearest")
        x = torch.cat([x, c4], dim=1)
        out2 = self.detect2(x)                                    # 256 ch, stride 16

        x = self.conv2(out2)
        x = F.interpolate(x, size=c3.shape[-2:], mode="nearest")
        x = torch.cat([x, c3], dim=1)
        out3 = self.detect3(x)                                    # 128 ch, stride 8

        return out1, out2, out3  # coarse -> fine, matches convs_bridge/convs_pred order


# --------------------------------------------------------------------------
# Head: YOLOV3Head (per-scale 3x3 bridge conv + 1x1 prediction conv)
# --------------------------------------------------------------------------
class YOLOV3Head(nn.Module):
    def __init__(self, num_classes=1, num_anchors=3, in_channels=(512, 256, 128), out_channels=(1024, 512, 256)):
        super().__init__()
        pred_channels = num_anchors * (5 + num_classes)  # 5 = tx,ty,tw,th,obj
        self.convs_bridge = nn.ModuleList(
            [ConvModule(c_in, c_out, kernel_size=3, padding=1) for c_in, c_out in zip(in_channels, out_channels)]
        )
        self.convs_pred = nn.ModuleList(
            [nn.Conv2d(c_out, pred_channels, kernel_size=1) for c_out in out_channels]
        )
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, feats):
        preds = []
        for feat, bridge, pred in zip(feats, self.convs_bridge, self.convs_pred):
            preds.append(pred(bridge(feat)))
        return preds  # list of (N, num_anchors*(5+nc), H, W), coarse -> fine


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
class YOLOV3(nn.Module):
    def __init__(
        self,
        num_classes=1,
        anchor_base_sizes=(
            ((116, 90), (156, 198), (373, 326)),  # stride 32 (coarsest / detect1)
            ((30, 61), (62, 45), (59, 119)),       # stride 16
            ((10, 13), (16, 30), (33, 23)),        # stride 8 (finest / detect3)
        ),
        strides=(32, 16, 8),
        image_mean=(0.0, 0.0, 0.0),
        image_std=(255.0, 255.0, 255.0),
        conf_thresh=0.005,
        nms_thresh=0.45,
        max_per_img=100,
    ):
        super().__init__()
        self.backbone = Darknet53()
        self.neck = YOLOV3Neck()
        self.bbox_head = YOLOV3Head(num_classes=num_classes)

        self.register_buffer("image_mean", torch.tensor(image_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std).view(-1, 1, 1), persistent=False)

        self.anchor_base_sizes = anchor_base_sizes
        self.strides = strides
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.max_per_img = max_per_img

    def normalize(self, img):
        return (img - self.image_mean) / self.image_std

    def _decode_level(self, pred, anchors, stride, device):
        """pred: (num_anchors*(5+nc), H, W) for a single image/level."""
        na = len(anchors)
        nc = self.num_classes
        H, W = pred.shape[-2:]
        pred = pred.view(na, 5 + nc, H, W).permute(0, 2, 3, 1)  # (na, H, W, 5+nc)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)  # (na, 2)

        tx, ty = pred[..., 0].sigmoid(), pred[..., 1].sigmoid()
        tw, th = pred[..., 2], pred[..., 3]
        obj = pred[..., 4].sigmoid()
        cls = pred[..., 5:].sigmoid()  # mmdet YOLOv3 uses independent sigmoid per class

        bx = (tx + grid_x.unsqueeze(0)) * stride
        by = (ty + grid_y.unsqueeze(0)) * stride
        bw = anchors_t[:, 0].view(na, 1, 1) * tw.exp()
        bh = anchors_t[:, 1].view(na, 1, 1) * th.exp()

        x1 = bx - bw / 2
        y1 = by - bh / 2
        x2 = bx + bw / 2
        y2 = by + bh / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
        scores = (obj.unsqueeze(-1) * cls).reshape(-1, nc)  # per-class confidence
        return boxes, scores

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
        scale_factor = min(416/H, 416/W)
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
        feats = self.neck(c3, c4, c5)
        preds = self.bbox_head(feats)  # coarse -> fine, matches self.strides/anchor order

        all_boxes, all_scores, all_labels = [], [], []
        for pred, anchors, stride in zip(preds, self.anchor_base_sizes, self.strides):
            boxes, scores = self._decode_level(pred[0], anchors, stride, device)
            for cls_idx in range(self.num_classes):
                s = scores[:, cls_idx]
                keep = s > self.conf_thresh
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
