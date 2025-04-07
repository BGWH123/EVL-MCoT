# Cross-Modal Hateful Meme Detection Framework

A state-of-the-art deep learning framework for detecting hateful content in multimodal meme data using Vision Transformer (ViT) and BERT.

## Overview

The framework combines computer vision and natural language processing to detect hateful content in memes. It leverages:

- Vision Transformer (ViT) for image feature extraction
- BERT for text feature extraction
- Custom transformer-based fusion mechanism
- Advanced classification head

## Project Structure

```
vit_bert/
├── config.py           # Configuration management
├── model.py           # Model architecture definitions
├── dataset.py         # Data processing and loading
├── trainer.py         # Training and evaluation logic
├── main.py           # Main training script
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Features

- Advanced cross-modal feature extraction
- Efficient data processing pipeline
- Comprehensive logging and monitoring
- Model checkpointing
- Performance evaluation
- Deterministic training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vit_bert.git
cd vit_bert
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure paths in `config.py`

2. Run training:
```bash
python main.py
```

## Model Architecture

The framework consists of three main components:

1. **FeatureExtractor**: Extracts visual and textual features using ViT and BERT
2. **CrossModalTransformer**: Fuses multimodal features using attention mechanism
3. **CrossModalClassifier**: Classifies the fused features

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended
- See requirements.txt for detailed dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{vitbert2023,
  title={Cross-Modal Hateful Meme Detection using Vision Transformer and BERT},
  author={Your Name},
  journal={Journal of Machine Learning},
  year={2023}
}
``` 