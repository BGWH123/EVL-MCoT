import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from config import Config

class PrototypeGuidedPatchDecoder(nn.Module):
    def __init__(self, num_prototypes: int = Config.ARCHITECTURE.NUM_PROTOTYPES, 
                 feat_dim: int = Config.ARCHITECTURE.FEATURE_DIM):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feat_dim))
        self.Wa = nn.Linear(feat_dim, feat_dim, bias=False)
        self.Wb = nn.Linear(feat_dim, 1, bias=False)
        self.Wv = nn.Linear(feat_dim, feat_dim, bias=False)
        self.Wc = nn.Linear(feat_dim, feat_dim, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            feat_dim, 
            num_heads=Config.ARCHITECTURE.NUM_HEADS, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(feat_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = patch_features.shape
        protos = self.prototypes.unsqueeze(0).repeat(B, 1, 1)
        
        attn_output, _ = self.cross_attn(query=protos, key=patch_features, value=patch_features)
        attn_weights = F.softmax(attn_output, dim=-1)
        attn_output = self.norm(attn_weights)
        
        pr = attn_output + protos
        pr_prime = self.Wa(pr)
        attention_scores = self.Wb(torch.tanh(self.Wv(pr_prime)))
        attention_weights = F.softmax(attention_scores, dim=1)
        slide_features = self.Wc(torch.sum(attention_weights * pr_prime, dim=1))
        
        return pr, slide_features

class ContextGuidedTextDecoder(nn.Module):
    def __init__(self, feat_dim: int = Config.ARCHITECTURE.FEATURE_DIM):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            feat_dim, 
            num_heads=Config.ARCHITECTURE.NUM_HEADS, 
            batch_first=True
        )

    def forward(self, text_features: torch.Tensor, image_context: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.cross_attn(
            query=image_context,
            key=text_features,
            value=text_features
        )
        attn_weights = F.softmax(attn_output, dim=-1)
        attn_weights += image_context
        return attn_output

class TextEnhancedCLIP(nn.Module):
    def __init__(self, clip_model_path: str):
        super().__init__()
        import sys
        sys.path.append("models/Long-CLIP")
        from model import longclip
        
        self.model, self.preprocess = longclip.load(clip_model_path)
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.patch_decoder = PrototypeGuidedPatchDecoder()
        self.hateful_text_decoder = ContextGuidedTextDecoder()
        self.benign_text_decoder = ContextGuidedTextDecoder()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_ce = nn.CrossEntropyLoss()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, 
        pixel_values: torch.Tensor,
        hateful_input_ids: torch.Tensor,
        benign_input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_output = self.model.encode_image(pixel_values)
        pr, slide_features = self.patch_decoder(vision_output)
        
        _, hateful_features = self.model.encode_text(hateful_input_ids)
        _, benign_features = self.model.encode_text(benign_input_ids)
        
        hate_enhanced = self.hateful_text_decoder(hateful_features, pr)
        benign_enhanced = self.benign_text_decoder(benign_features, pr)
        
        hate_enhanced = F.avg_pool2d(hate_enhanced.unsqueeze(1), kernel_size=(16, 1)).squeeze(1)
        benign_enhanced = F.avg_pool2d(benign_enhanced.unsqueeze(1), kernel_size=(16, 1)).squeeze(1)
        
        text_features = torch.cat([benign_enhanced, hate_enhanced], dim=1)
        
        slide_features = slide_features / slide_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        sim = torch.einsum("bnd,bd->bn", text_features, slide_features)
        logits = self.logit_scale * sim
        
        if labels is not None:
            loss = self.loss_ce(logits, labels)
            return loss, logits
        return logits

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v.contiguous() if isinstance(v, torch.Tensor) else v 
                for k, v in state_dict.items()} 