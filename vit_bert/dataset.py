import os
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTImageProcessor
from config import Config

class MemeDataProcessor:
    def __init__(self, csv_path: Path, image_dir: Path):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.data = self._load_and_process_data()
        
    def _load_and_process_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=["id", "text", "hateful_1", "hateful_2","hatefuL_3", "benign_1", "benign_2", "benign_3","label"])
        return df
        
    def generate_pairs(self) -> List[Tuple[str, str, str, str, int]]:
        pairs = []
        for _, row in self.data.iterrows():
            img_path = self.image_dir / f"{str(row['id']).zfill(5)}.png"
            pairs.extend([
                (img_path, row["text"], row["hateful_1"], row["benign_1"], row["label"]),
                (img_path, row["text"], row["hateful_1"], row["benign_2"], row["label"]),
                (img_path, row["text"], row["hateful_1"], row["benign_3"], row["label"]),
                (img_path, row["text"], row["hateful_2"], row["benign_1"], row["label"]),
                (img_path, row["text"], row["hateful_2"], row["benign_2"], row["label"]),
                (img_path, row["text"], row["hateful_2"], row["benign_3"], row["label"]),
                (img_path, row["text"], row["hateful_3"], row["benign_1"], row["label"]),
                (img_path, row["text"], row["hateful_3"], row["benign_2"], row["label"]),
                (img_path, row["text"], row["hateful_3"], row["benign_3"], row["label"]),
            ])
        return pairs

class MemeFeatureExtractor:
    def __init__(self, vit_processor: ViTImageProcessor, bert_tokenizer: BertTokenizer):
        self.vit_processor = vit_processor
        self.bert_tokenizer = bert_tokenizer
        
    def process_image(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        return self.vit_processor(images=image, return_tensors="pt")
        
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        return self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

class HatefulMemeDataset(Dataset):
    def __init__(self, data_processor: MemeDataProcessor, feature_extractor: MemeFeatureExtractor):
        self.data_processor = data_processor
        self.feature_extractor = feature_extractor
        self.pairs = self.data_processor.generate_pairs()
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, context, hateful_text, benign_text, label = self.pairs[idx]
        
        image = Image.open(img_path).convert("RGB")
        image_inputs = self.feature_extractor.process_image(image)
        
        hateful_inputs = self.feature_extractor.process_text(f"{context} {hateful_text}")
        benign_inputs = self.feature_extractor.process_text(f"{context} {benign_text}")
        
        return {
            "image_features": image_inputs,
            "hateful_text_features": hateful_inputs,
            "benign_text_features": benign_inputs,
            "labels": torch.tensor(label)
        }

class DataCollator:
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "image_features": torch.stack([item["image_features"]["pixel_values"].squeeze(0) for item in batch]),
            "hateful_text_features": {k: torch.stack([item["hateful_text_features"][k].squeeze(0) for item in batch]) 
                                    for k in batch[0]["hateful_text_features"].keys()},
            "benign_text_features": {k: torch.stack([item["benign_text_features"][k].squeeze(0) for item in batch]) 
                                   for k in batch[0]["benign_text_features"].keys()},
            "labels": torch.stack([item["labels"] for item in batch])
        } 