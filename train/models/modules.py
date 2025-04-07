import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..config import Config

class MultiHeadAttention(nn.Module):
    """Multi-head attention module with improved implementation."""
    
    def __init__(
        self,
        hidden_size: int = Config.ARCHITECTURE.HIDDEN_SIZE,
        num_attention_heads: int = Config.ARCHITECTURE.NUM_ATTENTION_HEADS,
        dropout_prob: float = Config.ARCHITECTURE.ATTENTION_PROBS_DROPOUT_PROB,
        position_embedding_type: str = "relative"  
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} is not divisible by number of attention heads {num_attention_heads}"
            )
            
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.position_embedding_type = position_embedding_type
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
   
        self.output = nn.Linear(hidden_size, hidden_size)
        

        if position_embedding_type == "relative":
            self.relative_positions_encoding = nn.Parameter(
                torch.randn(2 * Config.DATA.MAX_TEXT_LENGTH - 1, self.attention_head_size)
            )
            
        self.dropout = nn.Dropout(dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_layer = self.transpose_for_scores(self.query(query_states))
        key_layer = self.transpose_for_scores(self.key(key_states))
        value_layer = self.transpose_for_scores(self.value(value_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative":
            seq_length = query_states.size(1)
            position_ids_l = torch.arange(seq_length, device=query_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, device=query_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r + seq_length - 1
            positional_embedding = self.relative_positions_encoding[distance]
            relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores
        
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        

        output = self.output(context_layer)
        
        return output, attention_probs

class TransformerLayer(nn.Module):
    """Transformer layer with improved implementation."""
    
    def __init__(
        self,
        hidden_size: int = Config.ARCHITECTURE.HIDDEN_SIZE,
        intermediate_size: int = Config.ARCHITECTURE.INTERMEDIATE_SIZE,
        num_attention_heads: int = Config.ARCHITECTURE.NUM_ATTENTION_HEADS,
        dropout_prob: float = Config.ARCHITECTURE.HIDDEN_DROPOUT_PROB,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads)

        self.intermediate = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(intermediate_size, hidden_size)
        )
        

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        normed_hidden_states = self.norm1(hidden_states)
        attention_output, attention_probs = self.attention(
            normed_hidden_states, normed_hidden_states, normed_hidden_states, attention_mask
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        

        normed_hidden_states = self.norm2(hidden_states)
        intermediate_output = self.intermediate(normed_hidden_states)
        layer_output = hidden_states + self.dropout(intermediate_output)
        
        return layer_output, attention_probs

class CrossModalEncoder(nn.Module):
    """Cross-modal encoder with improved implementation."""
    
    def __init__(
        self,
        num_hidden_layers: int = Config.ARCHITECTURE.NUM_HIDDEN_LAYERS,
        hidden_size: int = Config.ARCHITECTURE.HIDDEN_SIZE
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size) for _ in range(num_hidden_layers)
        ])
        self.gradient_checkpointing = False
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_attentions = []
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask
                )
            else:
                layer_outputs = layer(hidden_states, attention_mask)
                
            hidden_states, attention_probs = layer_outputs
            all_attentions.append(attention_probs)
            
        return hidden_states, torch.stack(all_attentions)

class ProjectionHead(nn.Module):
    """Improved projection head with residual connections."""
    
    def __init__(
        self,
        input_dim: int = Config.MODEL.FEATURE_DIM,
        hidden_dim: int = Config.MODEL.PROJECTION_DIM,
        output_dim: int = Config.ARCHITECTURE.NUM_CLASSES,
        dropout: float = Config.MODEL.DROPOUT_RATE
    ):
        super().__init__()
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj3 = nn.Linear(hidden_dim, output_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        projected = self.proj1(x)
        projected = self.norm1(projected)
        projected = self.activation(projected)
        projected = self.dropout(projected)
        

        residual = projected
        projected = self.proj2(projected)
        projected = self.norm2(projected)
        projected = self.activation(projected)
        projected = self.dropout(projected)
        projected = projected + residual

        return self.proj3(projected) 