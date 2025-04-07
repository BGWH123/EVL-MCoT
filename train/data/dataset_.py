import pandas as pd
import os
import json
from PIL import Image
from typing import List, Dict, Any, Optional
from pathlib import Path

class MemeDataLoader:
    """
    A data loader class for handling meme datasets.
    Supports loading and processing of meme images and their associated metadata.
    """
    
    def __init__(self, dataset_path: str, name: str, splits: str):
        """
        Initialize the MemeDataLoader.
        
        Args:
            dataset_path (str): Path to the dataset directory
            name (str): Name of the dataset
            splits (str): Dataset split identifier
        """
        self.dataset_path = Path(dataset_path)
        self.name = name
        self.split = splits
        if name == 'hateful_memes':
            self.meme_path = self.dataset_path / 'hateful_memes'

    def load_meme_data(self, split: str) -> List[Dict[str, Any]]:
        """
        Load meme data for the specified split.
        
        Args:
            split (str): Dataset split to load (train, dev_seen, test_seen, etc.)
            
        Returns:
            List[Dict[str, Any]]: List of meme data entries containing:
                - id: Unique identifier
                - img: PIL Image object
                - text: Associated text
                - label: Classification label
        """
        meme_data = []
        for filename in os.listdir(self.meme_path):
            if filename.startswith(split) and filename.endswith('.jsonl'):
                file_path = self.meme_path / filename
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        img_path = self.meme_path / item["img"]
                        try:
                            img = Image.open(img_path).convert("RGB")
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
                            img = None
                        meme_data.append({
                            "id": item["id"],
                            "img": img,
                            "text": item["text"],
                            "label": item["label"]
                        })
        return meme_data

def test_loader() -> None:
    """
    Test function for the MemeDataLoader class.
    """
    dataset_path = "data"
    name = 'hateful_memes'
    splits = 'train'
    meme_data_loader = MemeDataLoader(dataset_path, name, splits)
    meme_data = meme_data_loader.load_meme_data('train')
    print(meme_data[0])

if __name__ == "__main__":
    test_loader()