import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import sys
sys.path.append("models/Long-CLIP")
from model import longclip
from config import Config

class MemeDataProcessor:
    def __init__(self, csv_path: Path, image_dir: Path):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.data = self._load_and_process_data()
        
    def _load_and_process_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, encoding="ISO-8859-1")
        required_columns = [
            "image_name", "sentence", "label", 
            "offensive_1", "offensive_2", "offensive_3",
            "no_offensive_1", "no_offensive_2", "no_offensive_3"
        ]
        return df.dropna(subset=required_columns)
        
    def generate_pairs(self) -> List[Tuple[str, str, str, str, int]]:
        pairs = []
        for _, row in self.data.iterrows():
            img_path = self.image_dir / row['image_name']
            pairs.extend([
                (img_path, row['sentence'], row["offensive_1"], row["no_offensive_1"], int(row["label"])),
                (img_path, row['sentence'], row["offensive_1"], row["no_offensive_2"], int(row["label"])),
                (img_path, row['sentence'], row["offensive_1"], row["no_offensive_3"], int(row["label"])),
                (img_path, row['sentence'], row["offensive_2"], row["no_offensive_1"], int(row["label"])),
                (img_path, row['sentence'], row["offensive_2"], row["no_offensive_2"], int(row["label"])),
                (img_path, row['sentence'], row["offensive_2"], row["no_offensive_3"], int(row["label"])),
                (img_path, row['sentence'], row["offensive_3"], row["no_offensive_1"], int(row["label"])),
                (img_path, row['sentence'], row["offensive_3"], row["no_offensive_2"], int(row["label"])),
                (img_path, row['sentence'], row["offensive_3"], row["no_offensive_3"], int(row["label"]))
            ])
        return pairs

class MemeFeatureExtractor:
    def __init__(self, clip_processor):
        self.clip_processor = clip_processor
        
    def process_image(self, image: Image.Image) -> torch.Tensor:
        return self.clip_processor(image)
        
    def process_text(self, text: str) -> torch.Tensor:
        return longclip.tokenize(
            text, 
            context_length=Config.ARCHITECTURE.CONTEXT_LENGTH,
            truncate=True
        )

class MultiOFFMemeDataset(Dataset):
    def __init__(self, data_processor: MemeDataProcessor, feature_extractor: MemeFeatureExtractor):
        self.data_processor = data_processor
        self.feature_extractor = feature_extractor
        self.pairs = self.data_processor.generate_pairs()
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, context, hateful_text, benign_text, label = self.pairs[idx]
        
        image = Image.open(img_path).convert("RGB")
        clip_inputs = self.feature_extractor.process_image(image)
        hateful_inputs = self.feature_extractor.process_text(hateful_text)
        benign_inputs = self.feature_extractor.process_text(benign_text)
        
        return {
            "pixel_values": clip_inputs.squeeze(0),
            "hateful_input_ids": hateful_inputs.squeeze(0),
            "benign_input_ids": benign_inputs.squeeze(0),
            "labels": torch.tensor(label)
        }

class DataCollator:
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "hateful_input_ids": torch.stack([item["hateful_input_ids"] for item in batch]),
            "benign_input_ids": torch.stack([item["benign_input_ids"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch])
        }

def test_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = Path("data/multiOFF/Train_clear.csv")
    image_dir = Path("data/MultiOFF/Labelled_Images")
    clip_path = Path("models/Long-CLIP/checkpoints/longclip-B.pt")
    
    _, preprocess = longclip.load(clip_path, device=device)
    data_processor = MemeDataProcessor(csv_path, image_dir)
    feature_extractor = MemeFeatureExtractor(preprocess)
    dataset = MultiOFFMemeDataset(data_processor, feature_extractor)
    sample = dataset[0]
    print(sample)

#test_dataset()
