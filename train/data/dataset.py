from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import sys
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
sys.path.append("mode/Long-CLIP")
from model import longclip
from ..config import Config

@dataclass
class InputExample:
    """A single training/test example for meme classification."""
    guid: str
    image_path: Path
    context: str
    hateful_text: str
    benign_text: str
    label: int

class DataProcessor:
    """Base class for data converters for meme classification data sets."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        
    def get_train_examples(self) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(
            self._read_csv(Config.DATA.TRAIN_CSV_PATH), "train"
        )
        
    def get_dev_examples(self) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(
            self._read_csv(Config.DATA.VAL_CSV_PATH), "dev"
        )
        
    def get_test_examples(self) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the test set."""
        return self._create_examples(
            self._read_csv(Config.DATA.TEST_CSV_PATH), "test"
        )
        
    def _read_csv(self, input_file: Path) -> pd.DataFrame:
        """Reads a CSV file."""
        return pd.read_csv(input_file, encoding="ISO-8859-1")
        
    def _create_examples(self, df: pd.DataFrame, set_type: str) -> List[InputExample]:
        """Creates examples for the training, dev and test sets."""
        examples = []
        for idx, row in df.iterrows():
            guid = f"{set_type}-{idx}"
            image_path = Config.DATA.IMAGE_DIR / f"{int(row['id']):05d}.png"
            examples.extend([
                InputExample(
                    guid=f"{guid}-{i}",
                    image_path=image_path,
                    context=row["text"],
                    hateful_text=row[f"hateful_{j}"],
                    benign_text=row[f"benign_{k}"],
                    label=int(row["label"])
                )
                for i, (j, k) in enumerate([(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)])
            ])
        return examples

class FeatureExtractor:
    """Extracts features from images and text using CLIP."""
    
    def __init__(self, clip_processor):
        self.clip_processor = clip_processor
        
    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Processes an image using CLIP processor."""
        return self.clip_processor(image)
        
    def process_text(self, text: str) -> torch.Tensor:
        """Processes text using CLIP tokenizer."""
        return longclip.tokenize(
            text,
            context_length=Config.DATA.MAX_TEXT_LENGTH,
            truncate=True
        )

class MemeDataset(Dataset):
    """Meme dataset for training and evaluation."""
    
    def __init__(
        self,
        examples: List[InputExample],
        feature_extractor: FeatureExtractor,
        max_length: Optional[int] = None
    ):
        self.examples = examples
        self.feature_extractor = feature_extractor
        self.max_length = max_length or Config.DATA.MAX_TEXT_LENGTH
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Load and process image
        image = Image.open(example.image_path).convert("RGB")
        image_features = self.feature_extractor.process_image(image)
        
        # Process texts
        hateful_features = self.feature_extractor.process_text(example.hateful_text)
        benign_features = self.feature_extractor.process_text(example.benign_text)
        
        return {
            "pixel_values": image_features.squeeze(0),
            "hateful_input_ids": hateful_features.squeeze(0),
            "benign_input_ids": benign_features.squeeze(0),
            "labels": torch.tensor(example.label)
        }

class DataCollator:
    """Data collator for batching."""
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "hateful_input_ids": torch.stack([f["hateful_input_ids"] for f in features]),
            "benign_input_ids": torch.stack([f["benign_input_ids"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])
        }
        return batch 