import torch
import random
import numpy as np
from typing import Tuple
from config import Config
from model import TextEnhancedCLIP
from dataset import MemeDataProcessor, MemeFeatureExtractor, MultiOFFMemeDataset
from trainer import setup_training, train_model

def set_seed(seed: int = Config.TRAINING.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_models() -> Tuple[TextEnhancedCLIP, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextEnhancedCLIP(str(Config.MODEL.CLIP_MODEL_PATH))
    model.to(device)
    return model, device

def setup_data() -> Tuple[MultiOFFMemeDataset, MultiOFFMemeDataset]:
    train_processor = MemeDataProcessor(Config.DATA.TRAIN_CSV_PATH, Config.DATA.IMAGE_DIR)
    test_processor = MemeDataProcessor(Config.DATA.TEST_CSV_PATH, Config.DATA.IMAGE_DIR)
    
    import sys
    sys.path.append("models/Long-CLIP")
    from model import longclip
    _, preprocess = longclip.load(str(Config.MODEL.CLIP_MODEL_PATH))
    feature_extractor = MemeFeatureExtractor(preprocess)
    
    train_dataset = MultiOFFMemeDataset(train_processor, feature_extractor)
    test_dataset = MultiOFFMemeDataset(test_processor, feature_extractor)
    
    return train_dataset, test_dataset

def main():
    set_seed()
    model, device = setup_models()
    train_dataset, test_dataset = setup_data()
    trainer = setup_training(model, train_dataset, test_dataset)
    train_model(trainer)

if __name__ == "__main__":
    main() 