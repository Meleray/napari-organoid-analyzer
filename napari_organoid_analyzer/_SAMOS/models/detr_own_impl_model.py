import torch
import pytorch_lightning as pl
from torch import nn, Tensor
from napari_organoid_analyzer._SAMOS.transformer_layers import TransformerDecoder, MLP, PositionEmbeddingSine
from napari_organoid_analyzer._SAMOS.matcher import HungarianMatcher
from napari_organoid_analyzer._SAMOS.losses import SetCriterion
from typing import Union



class DetectionTransformer(nn.Module):
    def __init__(self, 
                 backbone_name,
                 set_cost_class: float = 1.0,
                 set_cost_bbox: float = 5.0,
                 set_cost_giou: float = 2.0,
                 max_epochs: int = 500,
                 num_queries: int = 200,
                 transformer_dim: int = 256,
                 nheads: int = 8,
                 dim_feedforward: int = 512,
                 num_layers_low_res: int = 6,
                 num_layers_medium_res: int = 0,
                 num_layers_high_res: int = 0,
                 dropout: float = 0.1,
                 pre_norm: bool = True,
                 bbox_loss_coef: float = 5.0,
                 giou_loss_coef: float = 2.0,
                 eos_coef: float = 0.1,
                 aux_loss: bool = False,
                 add_query_before_output = False,
                 scale_bb_before_sigmoid = False,
                 **kwargs
                ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_layers_low_res = num_layers_low_res
        self.num_layers_medium_res = num_layers_medium_res
        self.num_layers_high_res = num_layers_high_res
        self.add_query_before_output = add_query_before_output
        self.scale_bb_before_sigmoid = scale_bb_before_sigmoid
        self.backbone = nn.Identity()

        self.activation = nn.ReLU()

        # Define matcher and loss here
        self.matcher = HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou)

        weight_dict = {'loss_ce': 1, 'loss_bbox': bbox_loss_coef, 'loss_giou': giou_loss_coef}
        
        if aux_loss:
            aux_weight_dict = {}
            for i in range(num_layers_low_res - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # for loss computation
        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(num_classes=1, 
                                      matcher=self.matcher, 
                                      weight_dict=weight_dict,
                                      eos_coef=eos_coef, 
                                      losses=losses)
        
        self.max_epochs = max_epochs
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, transformer_dim)


        # Define positional embedding
        N_steps = transformer_dim // 2
        self.position_embedding = PositionEmbeddingSine(N_steps, normalize=True)


        # Convert concatenated low res features to transformer dim
        if self.backbone_name == 'FM_concat':
            self.low_res_feature_adaptor = nn.Conv2d(512, transformer_dim, kernel_size=1)

        # Define the transformer module
        self.transformer_decoder = TransformerDecoder(
            transformer_dim=transformer_dim,
            nheads=nheads,
            num_layers=num_layers_low_res,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pre_norm=pre_norm,
            return_intermediate=aux_loss, # if use auxiliary loss, must return intermediate outputs
        )


        
        if (self.backbone_name == 'SAM_large') or (
            (self.num_layers_medium_res == 0) and 
            (self.num_layers_high_res == 0)
        ):
            assert self.num_layers_medium_res == 0, self.num_layers_medium_res
            assert self.num_layers_high_res == 0, self.num_layers_high_res
            self.class_embed = nn.Linear(transformer_dim, 2) # Binary classification: Object or No object
            self.bbox_embed = MLP(transformer_dim, transformer_dim, 4, 3)

        elif self.backbone_name in ['SAM2_large', 'FM_concat']:
            self.low2medium_res = nn.Linear(transformer_dim, 64)

            if self.num_layers_medium_res > 0:
                transformer_dim_medium = 64
                self.position_embedding_medium = PositionEmbeddingSine(transformer_dim_medium // 2, normalize=True)
                self.transformer_decoder_medium = TransformerDecoder(
                    transformer_dim=transformer_dim_medium,
                    nheads=2,
                    num_layers=num_layers_medium_res,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    pre_norm=pre_norm,
                    return_intermediate=aux_loss,
                )

            self.medium2high_res = nn.Linear(64, 32)

            if self.num_layers_high_res > 0:
                transformer_dim_high = 32
                self.position_embedding_high = PositionEmbeddingSine(transformer_dim_high // 2, normalize=True)
                self.transformer_decoder_high = TransformerDecoder(
                    transformer_dim=transformer_dim_high,
                    nheads=1,
                    num_layers=num_layers_high_res,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    pre_norm=pre_norm,
                    return_intermediate=aux_loss,
                )

            self.class_embed = nn.Linear(32, 2) # Binary classification: Object or No object
            self.bbox_embed = MLP(32, transformer_dim, 4, 3)
        else:
            raise ValueError(self.backbone_name)

        # self.bn_boxes = nn.BatchNorm1d(num_features=4*self.num_queries, momentum=0.01)
        
        self.aux_loss = aux_loss  # Ensure aux_loss is stored
        # if self.aux_loss:
        #     self.bn_boxes_aux = nn.BatchNorm1d(num_features=4*self.num_queries*(self.num_layers_low_res-1), momentum=0.01)



    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


    def forward(self, image_embeddings, window_size=1024):
        """Predicts boxes and scores for a batch of image embeddings

        Args:
            image_embeddings:
                For SAM_large: a torch.Tensor of shape [B, 256, 64, 64].
                For SAM2_large: a tuple of 3 torch.Tensors of shape [B, 256, 64, 64], [B, 64, 128, 128], [B, 32, 256, 256].
                For FM_concat: a tuple of 3 torch.Tensors of shape [B, 512, 64, 64], [B, 64, 128, 128], [B, 32, 256, 256].
        """
        output = self.forward_images(image_embeddings)

        boxes = output['pred_boxes']
        scores = output['pred_logits'].softmax(dim=-1)
        scores = scores[:, :, 0]  # class 0 is organoids, 1 is background
        labels = torch.zeros(scores.shape, dtype=torch.int64, device=scores.device)

        # Transform boxes:  [cy cx h w] in [0, 1] range  -->  [x y x y] in [0, 1024] px
        boxes = torch.stack([
            boxes[:, :, 1] - (boxes[:, :, 3] / 2),
            boxes[:, :, 0] - (boxes[:, :, 2] / 2),
            boxes[:, :, 1] + (boxes[:, :, 3] / 2),
            boxes[:, :, 0] + (boxes[:, :, 2] / 2),
        ], dim=2) * window_size

        predictions = []
        for batch_idx in range(scores.shape[0]):
            predictions.append({'scores': scores[batch_idx],
                                'boxes': boxes[batch_idx],
                                'labels': labels[batch_idx]})

        return predictions

    def forward_images(self, image_embeddings: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        # print('mem 1:', torch.cuda.memory_allocated())

        if self.backbone_name=='SAM_large':
            low_res_feats = image_embeddings
        elif self.backbone_name=='SAM2_large':
            assert isinstance(image_embeddings, list), type(image_embeddings)
            low_res_feats = image_embeddings[0]
            medium_res_feats = image_embeddings[1]
            high_res_feats = image_embeddings[2]
        elif self.backbone_name=='FM_concat':
            assert isinstance(image_embeddings, list), type(image_embeddings)
            low_res_feats = self.activation(self.low_res_feature_adaptor(image_embeddings[0]))
            medium_res_feats = image_embeddings[1]
            high_res_feats = image_embeddings[2]
        else:
            raise ValueError(self.backbone_name)
        
        # print('mem 2:', torch.cuda.memory_allocated())
        del image_embeddings
        # print('mem 3:', torch.cuda.memory_allocated())

        device = low_res_feats.device
        batch_size = low_res_feats.size(0)

        # Low resolution transformer input
        query_embedding = self.query_embed.weight.to(device)
        query_embedding = query_embedding.unsqueeze(0).expand(batch_size, -1, -1) # Add batch dimension and expand
        target = torch.zeros_like(query_embedding) # bs x num_queries x transformer_dim (256)
        
        # print('mem 4:', torch.cuda.memory_allocated())

        pos_embedding = self.position_embedding(low_res_feats) # bs x 256 x 64 x 64
        pos_embedding = pos_embedding.flatten(2).permute(0, 2, 1) # bs x (64x64) x 256
        low_res_feats = low_res_feats.flatten(2).permute(0, 2, 1) # bs x (64x64) x 256

        # print('mem 5:', torch.cuda.memory_allocated())

        # Use the transformer module
        target = self.transformer_decoder(target, query_embedding, low_res_feats, pos_embedding)

        # print('mem 6:', torch.cuda.memory_allocated())

        if self.aux_loss:
            target_aux = target[:-1]
            target = target[-1:]
        target = target.squeeze(0)

        if (self.backbone_name == 'SAM_large') or ((self.num_layers_medium_res == 0) and 
                                                   (self.num_layers_high_res == 0)):
            # print('\n\n target', target[0, 0, :5, :3].detach().cpu().numpy())
            # print('\n\n query_embedding', query_embedding[0, :5, :3].detach().cpu().numpy())
            pass
        elif self.backbone_name in ['SAM2_large', 'FM_concat']:
            # Proceed decoding with medium and high res features
            target = self.activation(self.low2medium_res(target))
            query_embedding = self.activation(self.low2medium_res(query_embedding))
            if self.aux_loss:
                target_aux = self.activation(self.low2medium_res(target_aux))
            if self.num_layers_medium_res > 0:
                pos_embedding = self.position_embedding_medium(medium_res_feats) # bs x 256 x 64 x 64
                pos_embedding = pos_embedding.flatten(2).permute(0, 2, 1) # bs x (64x64) x 256
                medium_res_feats = medium_res_feats.flatten(2).permute(0, 2, 1) # bs x (64x64) x 256
                target = self.transformer_decoder_medium(target, query_embedding, medium_res_feats, pos_embedding)
                if self.aux_loss:
                    target_aux_added = target[:-1]
                    target_aux = torch.cat([target_aux, target_aux_added], dim=0)
                    target = target[-1:]
                target = target.squeeze(0)
                
            target = self.activation(self.medium2high_res(target))
            query_embedding = self.activation(self.medium2high_res(query_embedding))
            if self.aux_loss:
                target_aux = self.activation(self.medium2high_res(target_aux))
            if self.num_layers_high_res > 0:
                pos_embedding = self.position_embedding_high(high_res_feats) # bs x 256 x 64 x 64
                pos_embedding = pos_embedding.flatten(2).permute(0, 2, 1) # bs x (64x64) x 256
                high_res_feats = high_res_feats.flatten(2).permute(0, 2, 1) # bs x (64x64) x 256
                # print('\n\n target.shape', target.shape, '\n\n')
                # print('\n\n query_embedding.shape', query_embedding.shape, '\n\n')
                # print('\n\n high_res_feats.shape', high_res_feats.shape, '\n\n')
                # print('\n\n pos_embedding.shape', pos_embedding.shape, '\n\n')
                target = self.transformer_decoder_high(target, query_embedding, high_res_feats, pos_embedding)
                # print('\n\n target', target[0, 0, :5, :3].detach().cpu().numpy())
                # print('\n\n query_embedding', query_embedding[0, :5, :3].detach().cpu().numpy())
                if self.aux_loss:
                    target_aux_added = target[:-1]
                    target_aux = torch.cat([target_aux, target_aux_added], dim=0)
                    target = target[-1:]
                target = target.squeeze(0)
                
        else:
            raise ValueError(self.backbone_name)
        
        target = target.unsqueeze(0)

        # Feed transformer output into MLP to get class and bbox
        if self.add_query_before_output:
            outputs_class = self.class_embed(target + query_embedding.unsqueeze(0))
            bbox_logits: torch.Tensor = self.bbox_embed(target + query_embedding.unsqueeze(0))
        else:
            outputs_class = self.class_embed(target)
            bbox_logits: torch.Tensor = self.bbox_embed(target)
        
        # # Apply batch normalization before the sigmoid function to improve pretraining convergence
        # # Non-auxiliary bbox logits get their own normalization
        # _, B, Q, _ = bbox_logits.shape  # 1 x batch x num_queries x 4
        # print('\n\n bbox_logits before bn', bbox_logits[0, :2, 0, :].detach().cpu().numpy(), '\n')
        # bbox_logits = bbox_logits.reshape(B, Q*4)  # batch x (num_queries * 4)
        # bbox_logits = self.bn_boxes(bbox_logits)
        # bbox_logits = bbox_logits.reshape(1, B, Q, 4)  # aux_targets x batch x num_queries x 4
        # print('\n\n bbox_logits after bn', bbox_logits[0, :2, 0, :].detach().cpu().numpy(), '\n')
        
        if self.aux_loss:
            if self.add_query_before_output:
                outputs_class_aux = self.class_embed(target_aux + query_embedding.unsqueeze(0).expand(target_aux.shape[0], -1, -1, -1))
                bbox_logits_aux = self.bbox_embed(target_aux + query_embedding.unsqueeze(0).expand(target_aux.shape[0], -1, -1, -1))
            else:
                outputs_class_aux = self.class_embed(target_aux)
                bbox_logits_aux = self.bbox_embed(target_aux)

            # # Every query and auxiliary loss gets their own normalization
            # AUX, B, Q, _ = bbox_logits_aux.shape  # [(aux targets + 1) x batch x num_queries x 4]
            # bbox_logits_aux = bbox_logits_aux.permute(1, 0, 2, 3).reshape(B, AUX*Q*4)  # batch x (aux_targets * num_queries * 4)
            # bbox_logits_aux = self.bn_boxes_aux(bbox_logits_aux)
            # bbox_logits_aux = bbox_logits_aux.reshape(B, AUX, Q, 4).permute(1, 0, 2, 3)  # aux_targets x batch x num_queries x 4


            bbox_logits = torch.cat([bbox_logits_aux, bbox_logits], dim=0)
            outputs_class = torch.cat([outputs_class_aux, outputs_class], dim=0)

        # print('bbox_logits.shape', bbox_logits.shape)
        # print('outputs_class.shape', outputs_class.shape)

        # print('outputs_class', outputs_class)
        # print('bbox_logits', bbox_logits)
        # if self.scale_bb_before_sigmoid:
        if False:
            outputs_coord = (bbox_logits * 20).sigmoid()  # [cy cx h w] in [0, 1] range
        else:
            outputs_coord = bbox_logits.sigmoid()  # [cy cx h w] in [0, 1] range

        # print('class and coord shapes', outputs_class.shape, outputs_coord.shape)  # [1, 1, 100, 2], [1, 1, 100, 4]
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    def forward_batch(self, batch):
        image_embedding, targets = batch[0], batch[1]

        # Forward
        outputs = self.forward_images(image_embedding)
        
        # Process targets: filter out all-zero entries and create dictionary
        # targets are bounding boxes of shape [bs x num_queries x 4]
        processed_targets = []
        for target in targets:
            boxes = target['boxes']
            labels = target['labels']
            non_zero_indices = (boxes > -0.5).any(axis=1)
            # device = boxes.device
            
            # Transform boxes:  [x y x y] in [0, 1024] px  -->  [cy cx h w] in [0, 1] range
            boxes = torch.stack([
                (boxes[:, 1] + boxes[:, 3]) / 2,
                (boxes[:, 0] + boxes[:, 2]) / 2,
                boxes[:, 3] - boxes[:, 1],
                boxes[:, 2] - boxes[:, 0],
            ], dim=1) / 1024

            filtered_boxes = boxes[non_zero_indices]#.to(device)
            filtered_labels = labels[non_zero_indices]#.to(device)
            num_boxes = filtered_boxes.size(0)
            # print('num_boxes', num_boxes)
            processed_targets.append({
                'boxes': filtered_boxes,
                'labels': filtered_labels.to(dtype=torch.int64)  # Here, labels are 0 = ground truth, 1 = no object
            })
        
        # Loss
        loss_dict = self.criterion(outputs, processed_targets)
        weight_dict = self.criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # IoU
        giou = self.criterion.compute_giou(outputs, processed_targets)

        # Move IoU to CPU for logging purposes
        giou = giou.detach().cpu()

        return total_loss, loss_dict, {'giou': giou}
    

    def forward_train(self, batch):
        self.train()
        return self.forward_batch(batch)
    
    
    def forward_eval(self, batch):
        self.eval()
        return self.forward_batch(batch)

