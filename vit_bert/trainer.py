import os
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional
import wandb
from config import Config
from model import CrossModalFineTuner
from dataset import HatefulMemeDataset, DataCollator

class CrossModalTrainer(Trainer):
    def __init__(
        self,
        model: CrossModalFineTuner,
        train_dataset: HatefulMemeDataset,
        eval_dataset: Optional[HatefulMemeDataset] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollator(),
            **kwargs
        )
        
    def compute_loss(self, model, inputs, return_outputs=False):
        image_features = inputs["image_features"]
        hateful_text_features = inputs["hateful_text_features"]
        benign_text_features = inputs["benign_text_features"]
        labels = inputs["labels"]
        
        loss, outputs = model(
            image_features,
            hateful_text_features,
            benign_text_features,
            labels
        )
        
        return (loss, outputs) if return_outputs else loss
        
    def training_step(self, model, inputs):
        loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        return loss
        
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None):
        model = self.model
        model.eval()
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs in dataloader:
                loss = self.compute_loss(model, inputs)
                total_loss += loss.item() * inputs["labels"].size(0)
                total_samples += inputs["labels"].size(0)
                
        avg_loss = total_loss / total_samples
        metrics = {"eval_loss": avg_loss}
        
        return metrics

def setup_training(
    model: CrossModalFineTuner,
    train_dataset: HatefulMemeDataset,
    eval_dataset: Optional[HatefulMemeDataset] = None
) -> CrossModalTrainer:
    training_args = TrainingArguments(
        output_dir=str(Config.PATHS.OUTPUT_DIR),
        evaluation_strategy="epoch",
        learning_rate=Config.TRAINING.LEARNING_RATE,
        per_device_train_batch_size=Config.TRAINING.BATCH_SIZE,
        per_device_eval_batch_size=Config.TRAINING.EVAL_BATCH_SIZE,
        num_train_epochs=Config.TRAINING.NUM_EPOCHS,
        weight_decay=Config.TRAINING.WEIGHT_DECAY,
        logging_dir=str(Config.PATHS.LOGGING_DIR),
        report_to="wandb",
        logging_steps=Config.TRAINING.LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    trainer = CrossModalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    return trainer

def train_model(trainer: CrossModalTrainer):
    trainer.train()
    trainer.save_model(str(Config.PATHS.MODEL_SAVE_DIR)) 