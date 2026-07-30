
import cv2
import numpy as np
import torch
from typing import Literal
from contextlib import contextmanager
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(nn.Module):
    def __init__(self, 
                num_classes: int = 2, 
                decoder: Literal["FRCNN", "FRCNNv2"] = "FRCNNv2", 
                ):
        super().__init__()

        # Parameters for model initialization
        self.num_classes = num_classes
        self.decoder_name = decoder
     
        # Define the FasterRCNN net
        self.model = self.get_pretrained_FasterRCNN()
    
    def get_pretrained_FasterRCNN(self):
        # load a model pre-trained on COCO
        if self.decoder_name == "FRCNN":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                box_detections_per_img=200,
                weights_backbone="ResNet50_Weights.IMAGENET1K_V1"
                )
        elif self.decoder_name == "FRCNNv2":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                box_detections_per_img=200,
                weights_backbone="ResNet50_Weights.IMAGENET1K_V1"
                )
        else:
            raise ValueError(f'decoder {self.decoder_name}')
        
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes) 

        return model

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
        scale_factor = min(800/H, 800/W)
        H_new, W_new = int(H*scale_factor), int(W*scale_factor)
        image = cv2.resize(image, (H_new, W_new), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float)

        # Normalization
        image = (image - image.min()) / max(image.max() - image.min(), 1e-5)
        
        return [image], 1 / scale_factor

    # ---------------- forward (single image, inference only) ----------------
    @torch.no_grad()
    def forward(self, image, *, filter_class=None):
        """image: (H, W, 3) uint8 tensor."""
        image, scale_factor = self.preprocess(image)

        # Do a forward pass in FasterRCNN
        with self.set_training(False):  # Necessary to forward images without targets.
            detections = self.model(image)
            
        out = []
        for detection in detections:
            boxes = detection['boxes']
            labels = detection['labels']
            scores = detection['scores']

            if filter_class is not None:
                # Select only the results for one class (filter_class) 
                is_foreground = labels == filter_class
                boxes = boxes[is_foreground]
                labels = labels[is_foreground]
                scores = scores[is_foreground]
            if len(boxes) == 0:
                out.append({"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})
            else:
                out.append({'scores': scores, 'boxes': boxes * scale_factor, 'labels': labels})
        return out

    @contextmanager
    def set_training(self, training: bool):
        """
        Context manager to set the model to training or evaluation mode 
        and subsequently reset to previous state.
        """
        # remember the current training state
        model_training = self.model.training
        model_rpn_training = self.model.rpn.training
        model_roi_heads_training = self.model.roi_heads.training

        # set the model to the desired training state
        if training:
            self.model.train()
        else:
            self.model.eval()
        self.model.training = training
        self.model.rpn.training = training
        self.model.roi_heads.training = training

        try:
            yield
        finally:
            # reset the model to the previous training state
            if model_training:
                self.model.train()
            else:
                self.model.eval()
            self.model.training = model_training
            self.model.rpn.training = model_rpn_training
            self.model.roi_heads.training = model_roi_heads_training

