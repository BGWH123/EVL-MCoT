import torch
import torch.nn as nn
from transformers import ViTForImageClassification, BertModel
from typing import Dict, Optional, Tuple
from config import Config

class FeatureExtractor(nn.Module):
    def __init__(self, vit_model: ViTForImageClassification, bert_model: BertModel):
        super().__init__()
        self.vit_model = vit_model
        self.bert_model = bert_model
        
    def extract_image_features(self, image_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.vit_model(image_inputs).last_hidden_state
        
    def extract_text_features(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.bert_model(**text_inputs).last_hidden_state[:, 0, :]
        
class CrossModalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)

class CrossModalClassifier(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

class CrossModalFineTuner(nn.Module):
    def __init__(self, vit_model: ViTForImageClassification, bert_model: BertModel):
        super().__init__()
        self.feature_extractor = FeatureExtractor(vit_model, bert_model)
        self.transformer = CrossModalTransformer(
            d_model=Config.ARCHITECTURE.TRANSFORMER_D_MODEL,
            nhead=Config.ARCHITECTURE.TRANSFORMER_NHEAD
        )
        self.classifier = CrossModalClassifier(
            d_model=Config.ARCHITECTURE.TRANSFORMER_D_MODEL,
            num_classes=Config.ARCHITECTURE.NUM_CLASSES
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(
        self, 
        image_features: Dict[str, torch.Tensor],
        hateful_text_features: Dict[str, torch.Tensor],
        benign_text_features: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embeds = self.feature_extractor.extract_image_features(image_features)
        hateful_embeds = self.feature_extractor.extract_text_features(hateful_text_features)
        benign_embeds = self.feature_extractor.extract_text_features(benign_text_features)
        
        combined_embeds = torch.cat([image_embeds, hateful_embeds, benign_embeds], dim=1)
        transformer_out = self.transformer(combined_embeds)
        
        hateful_logits = self.classifier(transformer_out[:, 0, :])
        benign_logits = self.classifier(transformer_out[:, 1, :])
        
        logits = torch.stack([hateful_logits, benign_logits], dim=1)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits
        
    def state_dict(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v.contiguous() if isinstance(v, torch.Tensor) else v 
                for k, v in state_dict.items()} 