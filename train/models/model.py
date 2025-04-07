import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import sys
sys.path.append("mode/Long-CLIP")
from model import longclip
from .modules import CrossModalEncoder, ProjectionHead
from ..config import Config

class EnhancedCrossModalCLIP(nn.Module):
    """Enhanced cross-modal CLIP model with advanced attention mechanisms."""
    
    def __init__(self, clip_model_path: str):
        super().__init__()
        self.model, self.preprocess = longclip.load(clip_model_path)
        
        # Freeze CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Cross-modal encoder
        self.cross_modal_encoder = CrossModalEncoder()
        
        # Projection heads
        self.image_projection = ProjectionHead(
            input_dim=Config.MODEL.FEATURE_DIM,
            hidden_dim=Config.MODEL.PROJECTION_DIM,
            output_dim=Config.MODEL.FEATURE_DIM
        )
        self.text_projection = ProjectionHead(
            input_dim=Config.MODEL.FEATURE_DIM,
            hidden_dim=Config.MODEL.PROJECTION_DIM,
            output_dim=Config.MODEL.FEATURE_DIM
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(Config.MODEL.FEATURE_DIM * 2, Config.MODEL.PROJECTION_DIM),
            nn.LayerNorm(Config.MODEL.PROJECTION_DIM),
            nn.GELU(),
            nn.Dropout(Config.MODEL.DROPOUT_RATE),
            nn.Linear(Config.MODEL.PROJECTION_DIM, Config.ARCHITECTURE.NUM_CLASSES)
        )
        
        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for all trainable components."""
        def _init_module(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
        self.cross_modal_encoder.apply(_init_module)
        self.image_projection.apply(_init_module)
        self.text_projection.apply(_init_module)
        self.classifier.apply(_init_module)
        
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image using CLIP and project features."""
        with torch.no_grad():
            image_features = self.model.encode_image(pixel_values)
        image_features = self.image_projection(image_features)
        return image_features
        
    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode text using CLIP and project features."""
        with torch.no_grad():
            text_features = self.model.encode_text(input_ids)[1]  # Use pooled output
        text_features = self.text_projection(text_features)
        return text_features
        
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Compute similarity between image and text features."""
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return torch.matmul(image_features, text_features.transpose(-2, -1))
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        hateful_input_ids: torch.Tensor,
        benign_input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        # Extract features
        image_features = self.encode_image(pixel_values)
        hateful_features = self.encode_text(hateful_input_ids)
        benign_features = self.encode_text(benign_input_ids)
        
        # Combine features for cross-modal attention
        combined_features = torch.cat([image_features.unsqueeze(1), 
                                     hateful_features.unsqueeze(1),
                                     benign_features.unsqueeze(1)], dim=1)
        
        # Apply cross-modal attention
        attended_features, attention_probs = self.cross_modal_encoder(combined_features)
        
        # Extract attended features
        attended_image = attended_features[:, 0]
        attended_hateful = attended_features[:, 1]
        attended_benign = attended_features[:, 2]
        
        # Compute similarities
        hateful_sim = self.compute_similarity(attended_image, attended_hateful)
        benign_sim = self.compute_similarity(attended_image, attended_benign)
        
        # Scale similarities
        hateful_sim = self.logit_scale.exp() * hateful_sim
        benign_sim = self.logit_scale.exp() * benign_sim
        
        # Concatenate features for classification
        concat_features = torch.cat([attended_hateful, attended_benign], dim=-1)
        logits = self.classifier(concat_features)
        
        if not return_dict:
            return (logits, hateful_sim, benign_sim, attention_probs)
            
        outputs = {
            "logits": logits,
            "hateful_similarity": hateful_sim,
            "benign_similarity": benign_sim,
            "attention_probs": attention_probs,
            "image_features": attended_image,
            "hateful_features": attended_hateful,
            "benign_features": attended_benign
        }
        
        if labels is not None:
            # Classification loss
            cls_loss = self.ce_loss(logits, labels)
            
            # Contrastive loss
            contrastive_labels = F.one_hot(labels, num_classes=2).float()
            contrastive_loss = self.kl_loss(
                F.log_softmax(torch.stack([hateful_sim, benign_sim], dim=1), dim=1),
                contrastive_labels
            )
            
            # Total loss with weighting
            total_loss = cls_loss + 0.5 * contrastive_loss  # 
            
            outputs.update({
                "loss": total_loss,
                "cls_loss": cls_loss,
                "contrastive_loss": contrastive_loss
            })
            
        return outputs
        
    def generate_attention_maps(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Generate attention maps for visualization."""
        with torch.no_grad():
            image_features = self.encode_image(pixel_values)
            text_features = self.encode_text(input_ids)
            
            combined_features = torch.cat([
                image_features.unsqueeze(1),
                text_features.unsqueeze(1)
            ], dim=1)
            
            _, attention_maps = self.cross_modal_encoder(combined_features)
            
        return attention_maps
        
    def state_dict(self, *args, **kwargs):
        """Ensure all tensors are contiguous when saving."""
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v.contiguous() if isinstance(v, torch.Tensor) else v 
                for k, v in state_dict.items()} 