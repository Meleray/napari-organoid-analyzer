from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math

class CrossAttentionLayer(nn.Module):
    """
    Implements a cross-attention layer with optional pre-normalization.
    
    Parameters:
    - d_model (int): Dimensionality of the model.
    - nhead (int): Number of attention heads.
    - dropout (float): Dropout rate.
    - normalize_before (bool): Whether to apply normalization before the attention mechanism.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, normalize_before: bool = False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initializes parameters with Xavier uniform distribution for tensors with more than one dimension."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """
        Optionally adds positional embeddings to the tensor.
        
        Args:
        - tensor (Tensor): Input tensor.
        - pos (Optional[Tensor]): Positional tensor to be added to the input tensor, if not None.
        
        Returns:
        - Tensor: Modified tensor with positional embeddings added.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, target: Tensor, source: Tensor,
                     target_pos: Optional[Tensor] = None,
                     source_mask: Optional[Tensor] = None,
                     source_pos: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with post normalization.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - source (Tensor): Source sequence tensor used as key and value in attention.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target tensor.
        - source_mask (Optional[Tensor]): Optional mask for the source tensor.
        - source_pos (Optional[Tensor]): Optional positional embeddings for the source tensor.
        
        Returns:
        - Tensor: Output tensor after applying cross-attention and normalization.
        """
        target2 = self.multihead_attn(query=self.with_pos_embed(target, target_pos),
                                      key=self.with_pos_embed(source, source_pos),
                                      value=source, attn_mask=source_mask)[0]
        target = target + self.dropout(target2)
        target = self.norm(target)
        return target

    def forward_pre(self, target: Tensor, source: Tensor,
                    target_pos: Optional[Tensor] = None,
                    source_mask: Optional[Tensor] = None,
                    source_pos: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with pre normalization.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - source (Tensor): Source sequence tensor used as key and value in attention.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target tensor.
        - source_mask (Optional[Tensor]): Optional mask for the source tensor.
        - source_pos (Optional[Tensor]): Optional positional embeddings for the source tensor.
        
        Returns:
        - Tensor: Output tensor after applying normalization and cross-attention.
        """
        target2 = self.norm(target)
        target2 = self.multihead_attn(query=self.with_pos_embed(target2, target_pos),
                                      key=self.with_pos_embed(source, source_pos),
                                      value=source, attn_mask=source_mask)[0]
        target = target + self.dropout(target2)
        return target

    def forward(self, target: Tensor, source: Tensor,
                target_pos: Optional[Tensor] = None,
                source_mask: Optional[Tensor] = None,
                source_pos: Optional[Tensor] = None) -> Tensor:
        """
        Defines the forward pass with optional pre or post normalization.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - source (Tensor): Source sequence tensor.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target.
        - source_mask (Optional[Tensor]): Optional mask for the source.
        - source_pos (Optional[Tensor]): Optional positional embeddings for the source.
        
        Returns:
        - Tensor: Output tensor after processing through the attention mechanism.
        """
        if self.normalize_before:
            return self.forward_pre(target, source, target_pos, source_mask, source_pos)
        return self.forward_post(target, source, target_pos, source_mask, source_pos)

class SelfAttentionLayer(nn.Module):
    """
    Implements a self-attention layer with optional pre-normalization.
    
    Parameters:
    - d_model (int): Dimensionality of the model.
    - nhead (int): Number of attention heads.
    - dropout (float): Dropout rate.
    - normalize_before (bool): Whether to apply normalization before the attention mechanism.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """
        Initializes parameters with Xavier uniform distribution for tensors with more than one dimension.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """
        Optionally adds positional embeddings to the tensor.
        
        Args:
        - tensor (Tensor): Input tensor.
        - pos (Optional[Tensor]): Positional tensor to be added to the input tensor, if not None.
        
        Returns:
        - Tensor: Modified tensor with positional embeddings added.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, target: Tensor,
                 target_mask: Optional[Tensor] = None,
                 target_pos: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with post normalization.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - target_mask (Optional[Tensor]): Optional mask for the target tensor.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target tensor.
        
        Returns:
        - Tensor: Output tensor after applying self-attention and normalization.
        """
        
        q = k = self.with_pos_embed(target, target_pos)
        target2 = self.self_attn(q, k, value=target, attn_mask=target_mask)[0]
        target = target + self.dropout(target2)
        target = self.norm(target)
        
        return target


    def forward_pre(self, target: Tensor,
                    target_mask: Optional[Tensor] = None,
                    target_pos: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with pre normalization.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - target_mask (Optional[Tensor]): Optional mask for the target tensor.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target tensor.
        
        Returns:
        - Tensor: Output tensor after applying normalization and self-attention.
        """

        target2 = self.norm(target)
        q = k = self.with_pos_embed(target2, target_pos)
        target2 = self.self_attn(q, k, value=target2, attn_mask=target_mask)[0]
        target = target + self.dropout(target2)

        return target


    def forward(self, target: Tensor,
                target_mask: Optional[Tensor] = None,
                target_pos: Optional[Tensor] = None) -> Tensor:
        """
        Defines the forward pass with optional pre or post normalization based on configuration.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - target_mask (Optional[Tensor]): Optional mask for the target tensor.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target.
        
        Returns:
        - Tensor: Output tensor after processing through the self-attention mechanism.
        """
        if self.normalize_before:
            return self.forward_pre(target, target_mask, target_pos)
        return self.forward_post(target, target_mask, target_pos)

class FFNLayer(nn.Module):
    """
    Implements a feedforward neural network layer as used in transformers, with options for
    pre-normalization and various activations.
    
    Parameters:
    - d_model (int): Dimensionality of the model.
    - dim_feedforward (int): Dimensionality of the hidden layer.
    - dropout (float): Dropout rate.
    - activation (str): Type of activation function to use ('relu', 'gelu', 'glu').
    - normalize_before (bool): Whether to apply normalization before other operations.
    """
    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                 activation: str = "gelu", normalize_before: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initializes parameters with Xavier uniform distribution for tensors with more than one dimension.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self, target: Tensor) -> Tensor:
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target))))
        target = target + self.dropout(target2)
        target = self.norm(target)
        return target

    def forward_pre(self, target: Tensor) -> Tensor:
        target2 = self.norm(target)
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target2))))
        target = target + self.dropout(target2)
        return target

    def forward(self, target: Tensor) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(target)
        return self.forward_post(target)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape [batch_size, channels, height, width]

        Returns:
            pos: Positional encoding tensor of shape [batch_size, num_pos_feats*2, height, width]
        """
        batch_size, _, height, width = x.shape

        # Create position encodings
        y_embed = torch.arange(height, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, width)
        x_embed = torch.arange(width, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(height, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return pos

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_activation_fn(activation: str) -> callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"Activation should be 'relu', 'gelu', or 'glu', not {activation}.")

class TransformerDecoder(nn.Module):
    def __init__(self, transformer_dim, nheads, num_layers, dim_feedforward, dropout, pre_norm, return_intermediate):
        super().__init__()
        self.self_attention_layers = nn.ModuleList()
        self.cross_attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        self.return_intermediate = return_intermediate # if use auxiliary loss or not
        self.norm = nn.LayerNorm(transformer_dim)

        for _ in range(num_layers):
            self.self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=transformer_dim,
                    nhead=nheads,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
            self.cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=transformer_dim,
                    nhead=nheads,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )
            self.ffn_layers.append(
                FFNLayer(
                    d_model=transformer_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=pre_norm,
                )
            )

    def forward(self, target, query_embedding, image_embedding, pos_embedding):
        intermediate = []

        for self_attn, cross_attn, ffn in zip(self.self_attention_layers, self.cross_attention_layers, self.ffn_layers):
            # print('mem beginning of layer:', torch.cuda.memory_allocated())
            target = self_attn(target=target, target_pos=query_embedding)
            # print('mem beginning after self_attn:', torch.cuda.memory_allocated())
            target = cross_attn(target=target, target_pos=query_embedding, source=image_embedding, source_pos=pos_embedding)
            # print('mem beginning after cross_attn:', torch.cuda.memory_allocated())
            target = ffn(target=target)
            # print('mem beginning after ffn:', torch.cuda.memory_allocated())

            if self.return_intermediate:
                intermediate.append(self.norm(target))

        if self.norm is not None:
            target = self.norm(target)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(target)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return target.unsqueeze(0)