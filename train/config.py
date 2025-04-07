from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    CLIP_MODEL_PATH: Path = Path("mode/Long-CLIP/checkpoints/longclip-B.pt")
    FEATURE_DIM: int = 512
    PROJECTION_DIM: int = 256
    DROPOUT_RATE: float = 0.1
    
@dataclass
class DataConfig:
    """Configuration for data processing."""
    TRAIN_CSV_PATH: Path = Path("data/hateful_memes/train.csv")
    VAL_CSV_PATH: Path = Path("data/hateful_memes/val.csv")
    TEST_CSV_PATH: Path = Path("data/hateful_memes/test.csv")
    IMAGE_DIR: Path = Path("data/hateful_memes/img")
    MAX_TEXT_LENGTH: int = 77 * 4 - 60  # Long-CLIP max text length
    
@dataclass
class TrainingConfig:
    """Configuration for training process."""
    BATCH_SIZE: int = 16
    EVAL_BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 10
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.1
    LOGGING_STEPS: int = 50
    SEED: int = 42
    GRADIENT_ACCUMULATION_STEPS: int = 2
    MAX_GRAD_NORM: float = 1.0
    
@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture details."""
    NUM_ATTENTION_HEADS: int = 8
    NUM_HIDDEN_LAYERS: int = 6
    HIDDEN_SIZE: int = 512
    INTERMEDIATE_SIZE: int = 2048
    HIDDEN_DROPOUT_PROB: float = 0.1
    ATTENTION_PROBS_DROPOUT_PROB: float = 0.1
    NUM_CLASSES: int = 2
    
@dataclass
class PathConfig:
    """Configuration for file paths."""
    OUTPUT_DIR: Path = Path("outputs")
    LOGGING_DIR: Path = Path("logs")
    CACHE_DIR: Path = Path("cache")
    MODEL_SAVE_DIR: Path = Path("checkpoints")
    
class Config:
    """Main configuration class that combines all configs."""
    MODEL = ModelConfig()
    DATA = DataConfig()
    TRAINING = TrainingConfig()
    ARCHITECTURE = ModelArchitectureConfig()
    PATHS = PathConfig()
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert all configurations to a dictionary."""
        return {
            "model": {k: v for k, v in vars(cls.MODEL).items()},
            "data": {k: v for k, v in vars(cls.DATA).items()},
            "training": {k: v for k, v in vars(cls.TRAINING).items()},
            "architecture": {k: v for k, v in vars(cls.ARCHITECTURE).items()},
            "paths": {k: v for k, v in vars(cls.PATHS).items()}
        }
    
    @classmethod
    def create_directories(cls) -> None:
        """Create all necessary directories."""
        for path in [cls.PATHS.OUTPUT_DIR, cls.PATHS.LOGGING_DIR, 
                    cls.PATHS.CACHE_DIR, cls.PATHS.MODEL_SAVE_DIR]:
            path.mkdir(parents=True, exist_ok=True) 