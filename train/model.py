import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, BertModel
from typing import Dict, List, Optional, Tuple
from .config import ModelConfig

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class MemeClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(config.CLIP_MODEL_PATH)
        self.bert = BertModel.from_pretrained(config.BERT_MODEL_PATH)
        
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.image_proj = nn.Linear(config.FEATURE_DIM, config.PROJECTION_DIM)
        self.text_proj = nn.Linear(config.FEATURE_DIM, config.PROJECTION_DIM)
        
        self.transformer = TransformerEncoder(
            embed_dim=config.PROJECTION_DIM,
            num_heads=config.NUM_ATTENTION_HEADS,
            dropout=config.DROPOUT_RATE
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.PROJECTION_DIM, config.PROJECTION_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.PROJECTION_DIM // 2, config.NUM_CLASSES)
        )

    def forward(self, 
                pixel_values: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        text_features = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        
        image_features = self.image_proj(image_features)
        text_features = self.text_proj(text_features)
        
        features = torch.cat([image_features.unsqueeze(1), 
                            text_features.unsqueeze(1)], dim=1)
        
        features = self.transformer(features)
        logits = self.classifier(features.mean(dim=1))
        
        return logits 