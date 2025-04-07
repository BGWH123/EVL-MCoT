import torch
import random
import numpy as np
from typing import Tuple
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import BertModel, BertTokenizer
from config import Config
from model import CrossModalFineTuner
from dataset import MemeDataProcessor, MemeFeatureExtractor, HatefulMemeDataset
from trainer import setup_training, train_model

def set_seed(seed: int = Config.TRAINING.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_models() -> Tuple[ViTImageProcessor, ViTForImageClassification, BertTokenizer, BertModel, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize ViT
    vit_processor = ViTImageProcessor.from_pretrained(str(Config.MODEL.VIT_MODEL_PATH))
    vit_model = ViTForImageClassification.from_pretrained(str(Config.MODEL.VIT_MODEL_PATH))
    
    # Initialize BERT
    bert_tokenizer = BertTokenizer.from_pretrained(str(Config.MODEL.BERT_MODEL_PATH))
    bert_model = BertModel.from_pretrained(str(Config.MODEL.BERT_MODEL_PATH))
    
    # Move models to device
    vit_model.to(device)
    bert_model.to(device)
    
    return vit_processor, vit_model, bert_tokenizer, bert_model, device

def setup_data(
    vit_processor: ViTImageProcessor, 
    bert_tokenizer: BertTokenizer
) -> HatefulMemeDataset:
    # Initialize data processor
    data_processor = MemeDataProcessor(
        csv_path=Config.DATA.CSV_PATH,
        image_dir=Config.DATA.IMAGE_DIR
    )
    
    # Initialize feature extractor
    feature_extractor = MemeFeatureExtractor(vit_processor, bert_tokenizer)
    
    # Create dataset
    dataset = HatefulMemeDataset(data_processor, feature_extractor)
    
    return dataset

def main():
    set_seed()
    # Setup models and device
    vit_processor, vit_model, bert_tokenizer, bert_model, device = setup_models()
    
    # Setup data
    dataset = setup_data(vit_processor, bert_tokenizer)
    
    # Initialize model
    model = CrossModalFineTuner(vit_model, bert_model)
    model.to(device)
    
    # Setup trainer
    trainer = setup_training(model, dataset)
    
    # Train model
    train_model(trainer)

if __name__ == "__main__":
    main() 