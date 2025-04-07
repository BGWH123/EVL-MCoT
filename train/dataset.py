import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from typing import Tuple, Dict, Any

class MemeDataset(Dataset):
    """
    Dataset class for meme classification task.
    Handles loading and preprocessing of image-text pairs.
    """
    def __init__(self, csv_path: str, image_dir: str, processor: CLIPProcessor, max_length: int = 248):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to the CSV file containing meme data
            image_dir: Directory containing meme images
            processor: CLIP processor for image preprocessing
            max_length: Maximum length of text input
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        
        self.data = pd.read_csv(csv_path)
        self.image_paths = self.data['image'].tolist()
        self.texts = self.data['text'].tolist()
        self.labels = self.data['label'].tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing processed image and text data
        """
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        
        text = self.texts[idx]
        label = self.labels[idx]
        
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0).float()
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        
        return inputs 