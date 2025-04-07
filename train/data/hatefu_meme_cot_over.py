import csv
import os
import base64
from typing import List, Dict, Any
import openai
from dataset import multiDataLoader
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemeAnalyzer:
    def __init__(self, api_key: str, dataset: Any, output_csv: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.dataset = dataset
        self.output_csv = output_csv

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _generate_prompt(self, text: str, is_offensive: bool) -> str:
        role = "offensive meme analyzer" if is_offensive else "non-offensive meme analyzer"
        return f"""You are a professional {role}. Analyze the provided meme image and its text: "{text}".
        Provide a detailed explanation that:
        1. Describes the visual elements and their significance
        2. Explains the relationship between the image and text
        3. Analyzes the cultural/social context
        4. {"Explains why this meme is considered offensive" if is_offensive else "Explains why this meme is not offensive"}
        Be specific and provide concrete examples in your analysis."""

    def _analyze_meme(self, image_path: str, text: str, is_offensive: bool) -> str:
        base64_image = self._encode_image(image_path)
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._generate_prompt(text, is_offensive)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def load_existing_ids(self) -> set:
        existing_ids = set()
        if os.path.exists(self.output_csv):
            with open(self.output_csv, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader, None)
                for row in reader:
                    existing_ids.add(row[0])
        return existing_ids

    def process_dataset(self):
        existing_ids = self.load_existing_ids()
        
        with open(self.output_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if os.stat(self.output_csv).st_size == 0:
                writer.writerow([
                    "image_name", "sentence", "label",
                    "offensive_analysis_1", "offensive_analysis_2", "offensive_analysis_3",
                    "non_offensive_analysis_1", "non_offensive_analysis_2", "non_offensive_analysis_3"
                ])

            for i, data in enumerate(self.dataset):
                image_name = data['image_name']
                if image_name in existing_ids:
                    logger.info(f"Skipping {image_name}, already processed")
                    continue

                try:
                    image_path = str(Path(data['image_path']))
                    sentence = data['sentence']
                    label = data['label']

                    offensive_analyses = [
                        self._analyze_meme(image_path, sentence, True)
                        for _ in range(3)
                    ]
                    
                    non_offensive_analyses = [
                        self._analyze_meme(image_path, sentence, False)
                        for _ in range(3)
                    ]

                    writer.writerow([
                        image_name, sentence, label,
                        *offensive_analyses,
                        *non_offensive_analyses
                    ])
                    
                    logger.info(f"Processed sample {i + 1}: {image_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {image_name}: {str(e)}")
                    continue

        logger.info(f"Analysis completed. Results saved to {self.output_csv}")

def main():
    # Configuration
    API_KEY = "your-api-key-here"  # Replace with your OpenAI API key
    DATASET_PATH = "path/to/dataset"
    IMAGE_PATH = "path/to/images"
    OUTPUT_CSV = "meme_analysis_results.csv"
    
    # Initialize data loader
    data_loader = multiDataLoader(DATASET_PATH, IMAGE_PATH, "test")
    dataset = data_loader.load_meme_data("test")
    
    # Initialize and run analyzer
    analyzer = MemeAnalyzer(API_KEY, dataset, OUTPUT_CSV)
    analyzer.process_dataset()

if __name__ == "__main__":
    main()
