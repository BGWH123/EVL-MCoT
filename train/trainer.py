import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Mapping, Optional
from transformers import Trainer, TrainingArguments
import numpy as np
from .utils import compute_metrics, move_to_device

class MemeTrainer(Trainer):
    """
    Custom trainer for meme classification task.
    Handles model saving and data collation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
        
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = move_to_device(inputs, self.device)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            
            if prediction_loss_only:
                return (loss, None, None)
            
            logits = outputs.logits if isinstance(outputs, dict) else outputs[1]
            labels = inputs.get("labels")
            
            return (loss, logits, labels)
            
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        model = self.model
        model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for inputs in eval_dataloader:
            loss, logits, labels = self.prediction_step(model, inputs)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics[f"{metric_key_prefix}_loss"] = total_loss / len(eval_dataloader)
        
        return metrics

    def save_model(self, output_dir: str = None, _internal_call: bool = False) -> None:
        """
        Save the model ensuring all tensors are contiguous.
        
        Args:
            output_dir: Directory to save the model
            _internal_call: Whether this is an internal call
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Ensure all tensors are contiguous
        state_dict = self.model.state_dict()
        for key in state_dict:
            if not state_dict[key].is_contiguous():
                state_dict[key] = state_dict[key].contiguous()

        # Save model
        super().save_model(output_dir, _internal_call)
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

def default_data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom data collator for batch processing.
    
    Args:
        features: List of feature dictionaries
        
    Returns:
        Dictionary of batched features
    """
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Process labels
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Process other features
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            values = [f[k] for f in features if f[k] is not None]

            if len(values) == 0:
                continue

            if isinstance(values[0], torch.Tensor):
                max_len = max(v.shape[0] if v.ndimension() > 0 else 1 for v in values)
                padded_tensors = [
                    torch.cat([v, torch.zeros(max_len - v.shape[0], dtype=v.dtype)])
                    if v.ndimension() > 0 and v.shape[0] < max_len else v
                    for v in values
                ]
                batch[k] = torch.stack(padded_tensors)

            elif isinstance(values[0], np.ndarray):
                max_len = max(len(v) for v in values)
                padded_arrays = [
                    np.pad(v, (0, max_len - len(v)), mode='constant')
                    for v in values
                ]
                batch[k] = torch.tensor(np.stack(padded_arrays), dtype=torch.long)

            elif isinstance(values[0], list):
                max_len = max(len(v) for v in values)
                padded_lists = [
                    v + [0] * (max_len - len(v)) if len(v) < max_len else v
                    for v in values
                ]
                batch[k] = torch.tensor(padded_lists, dtype=torch.long)

            else:
                batch[k] = torch.tensor([v for v in values])

    return batch 