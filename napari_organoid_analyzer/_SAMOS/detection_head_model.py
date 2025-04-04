import torch
import pytorch_lightning as pl
from torch import nn, Tensor
from .transformer_layers import TransformerDecoder, MLP, PositionEmbeddingSine
from .matcher import HungarianMatcher
from .losses import SetCriterion

class DetectionHead(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-4, 
                weight_decay: float = 1e-4,
                lr_drop: float = 200,
                set_cost_class: float = 1.0,
                set_cost_bbox: float = 5.0,
                set_cost_giou: float = 2.0,
                max_epochs: int = 500,
                num_queries: int = 100,
                transformer_dim: int = 256,
                nheads: int = 8,
                dim_feedforward: int = 512,
                num_layers: int = 6,
                dropout: float = 0.1,
                pre_norm: bool = True,
                bbox_loss_coef: float = 5.0,
                giou_loss_coef: float = 2.0,
                eos_coef: float = 0.1,
                aux_loss: bool = False,
                ):
        super().__init__()
        self.save_hyperparameters()

        # Parameters for optimizer and scheduler
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_drop = lr_drop

        # Define matcher and loss here
        self.matcher = HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou)

        weight_dict = {'loss_ce': 1, 'loss_bbox': bbox_loss_coef, 'loss_giou': giou_loss_coef}
        
        if aux_loss:
            aux_weight_dict = {}
            for i in range(num_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # for loss computation
        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(num_classes=1, matcher=self.matcher, weight_dict=weight_dict,
                                eos_coef=eos_coef, losses=losses)
        
        self.max_epochs = max_epochs
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, transformer_dim)

        # Define positional embedding
        N_steps = transformer_dim // 2
        self.position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

        # Define the transformer module
        self.transformer_decoder = TransformerDecoder(
            transformer_dim=transformer_dim,
            nheads=nheads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pre_norm=pre_norm,
            return_intermediate=aux_loss, # if use auxiliary loss, must return intermediate outputs
        )

        self.class_embed = nn.Linear(transformer_dim, 2) # Binary classification: Object or No object
        self.bbox_embed = MLP(transformer_dim, transformer_dim, 4, 3)

        self.training_step_outputs = []
        self.val_step_outputs = []
        self.aux_loss = aux_loss  # Ensure aux_loss is stored

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(self, query_embedding, image_embedding, pos_embedding):
        batch_size = image_embedding.size(0)
        query_embedding = query_embedding.unsqueeze(0).expand(batch_size, -1, -1) # Add batch dimension and expand

        target = torch.zeros_like(query_embedding) # bs x 300 x 256
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1) # bs x (64x64) x 256
        pos_embedding = pos_embedding.flatten(2).permute(0, 2, 1) # bs x (64x64) x 256

        # Use the transformer module
        target = self.transformer_decoder(target, query_embedding, image_embedding, pos_embedding)

        # Feed transformer output into MLP to get class and bbox
        outputs_class = self.class_embed(target)
        outputs_coord = self.bbox_embed(target).sigmoid()
        # print('class and coord shapes', outputs_class.shape, outputs_coord.shape)  # [1, 1, 100, 2], [1, 1, 100, 4]
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_drop)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image_embedding, targets = batch
        device = image_embedding.device
        pos_embedding = self.position_embedding(image_embedding) # bs x 256 x 64 x 64

        # forward
        outputs = self.forward(query_embedding=self.query_embed.weight.to(device), image_embedding=image_embedding, pos_embedding=pos_embedding)
        
        # Process targets: filter out all-zero entries and create dictionary
        processed_targets = []
        for target in targets:
            non_zero_indices = ~(target == 0).all(axis=1)
            filtered_boxes = target[non_zero_indices].to(device)
            num_boxes = filtered_boxes.size(0)
            processed_targets.append({
                'boxes': filtered_boxes,
                'labels': torch.zeros(num_boxes, dtype=torch.int64, device=device)  # Here, labels are 0 = ground truth, 1 = no object
            })
        
        # Loss
        loss_dict = self.criterion(outputs, processed_targets)
        weight_dict = self.criterion.weight_dict
        train_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # IoU
        giou = self.criterion.compute_giou(outputs, processed_targets)

        # Move IoU to CPU for logging purposes
        giou = giou.detach().cpu()

        self.training_step_outputs.append({'loss': train_losses, 'giou': giou})

        return {'loss': train_losses}

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_train_iou = torch.stack([x['giou'] for x in self.training_step_outputs]).mean()

        self.log('train_loss', avg_train_loss)
        self.log('train_giou', avg_train_iou)

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        image_embedding, targets = batch
        device = image_embedding.device
        pos_embedding = self.position_embedding(image_embedding) # bs x 256 x 64 x 64
        outputs = self.forward(query_embedding=self.query_embed.weight.to(device), image_embedding=image_embedding, pos_embedding=pos_embedding)

        # Process targets: filter out all-zero entries and create dictionary
        processed_targets = []
        for target in targets:
            non_zero_indices = ~(target == 0).all(axis=1)
            filtered_boxes = target[non_zero_indices].to(device)
            num_boxes = filtered_boxes.size(0)
            processed_targets.append({
                'boxes': filtered_boxes,
                'labels': torch.zeros(num_boxes, dtype=torch.int64, device=device)  # All labels are 0
            })

        # Loss
        loss_dict = self.criterion(outputs, processed_targets)
        weight_dict = self.criterion.weight_dict
        val_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # IoU
        giou = self.criterion.compute_giou(outputs, processed_targets)

        # Move IoU to CPU for logging purposes
        giou = giou.detach().cpu()

        self.val_step_outputs.append({'loss': val_losses, 'giou': giou})

        return {'val_loss': val_losses}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
        avg_val_iou = torch.stack([x['giou'] for x in self.val_step_outputs]).mean()

        self.log('val_loss', avg_val_loss)
        self.log('val_giou', avg_val_iou)

        self.val_step_outputs.clear()
