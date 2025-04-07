import os
import torch
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional
import wandb
from config import Config
from model import TextEnhancedCLIP
from dataset import MultiOFFMemeDataset, DataCollator

class CustomTrainer(Trainer):
    def __init__(
        self,
        model: TextEnhancedCLIP,
        train_dataset: MultiOFFMemeDataset,
        eval_dataset: Optional[MultiOFFMemeDataset] = None,
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
        pixel_values = inputs["pixel_values"]
        hateful_input_ids = inputs["hateful_input_ids"]
        benign_input_ids = inputs["benign_input_ids"]
        labels = inputs["labels"]
        
        loss, outputs = model(
            pixel_values,
            hateful_input_ids,
            benign_input_ids,
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
        
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        state_dict = self.model.state_dict()
        
        for key in state_dict:
            if not state_dict[key].is_contiguous():
                state_dict[key] = state_dict[key].contiguous()

        super().save_model(output_dir, _internal_call)
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

def setup_training(
    model: TextEnhancedCLIP,
    train_dataset: MultiOFFMemeDataset,
    eval_dataset: Optional[MultiOFFMemeDataset] = None
) -> CustomTrainer:
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
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    return trainer

def train_model(trainer: CustomTrainer):
    trainer.train()
    trainer.save_model(str(Config.PATHS.MODEL_SAVE_DIR)) 