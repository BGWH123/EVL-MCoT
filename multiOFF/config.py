from pathlib import Path
from typing import Dict, Any

class ModelConfig:
    CLIP_MODEL_PATH = Path("models/Long-CLIP/checkpoints/longclip-B.pt")
    
class DataConfig:
    TRAIN_CSV_PATH = Path("data/multiOFF/Train.csv")
    TEST_CSV_PATH = Path("data/multiOFF/test.csv")
    IMAGE_DIR = Path("data/MultiOFF/Labelled_Images")
    
class TrainingConfig:
    BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 64
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WEIGHT_DECAY = 0.01
    LOGGING_STEPS = 500
    SEED = 42
    
class ModelArchitectureConfig:
    NUM_PROTOTYPES = 16
    FEATURE_DIM = 512
    NUM_HEADS = 8
    NUM_CLASSES = 2
    CONTEXT_LENGTH = 77 * 4 - 60
    
class PathConfig:
    OUTPUT_DIR = Path("outputs")
    LOGGING_DIR = Path("logs")
    MODEL_SAVE_DIR = Path("saved_models")
    
class Config:
    MODEL = ModelConfig()
    DATA = DataConfig()
    TRAINING = TrainingConfig()
    ARCHITECTURE = ModelArchitectureConfig()
    PATHS = PathConfig()
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            "model": {k: v for k, v in cls.MODEL.__dict__.items() if not k.startswith('_')},
            "data": {k: v for k, v in cls.DATA.__dict__.items() if not k.startswith('_')},
            "training": {k: v for k, v in cls.TRAINING.__dict__.items() if not k.startswith('_')},
            "architecture": {k: v for k, v in cls.ARCHITECTURE.__dict__.items() if not k.startswith('_')},
            "paths": {k: v for k, v in cls.PATHS.__dict__.items() if not k.startswith('_')}
        } 