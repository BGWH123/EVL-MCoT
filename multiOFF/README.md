# MultiOFF: Multimodal Offensive Content Detection Framework

A state-of-the-art deep learning framework for detecting offensive content in multimodal meme data using Long-CLIP and advanced attention mechanisms.

## Overview

MultiOFF is designed to detect offensive content in memes by analyzing both visual and textual components. The framework leverages:

- Long-CLIP for multimodal feature extraction
- Prototype-guided patch decoding
- Context-guided text enhancement
- Advanced attention mechanisms
- Sophisticated classification head

## Project Structure

```
multiOFF/
├── config.py           # Configuration management
├── model.py           # Model architecture definitions
├── dataset.py         # Data processing and loading
├── trainer.py         # Training and evaluation logic
├── main.py           # Main training script
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Features

- Advanced prototype-guided feature extraction
- Context-aware text processing
- Efficient data handling
- Comprehensive logging
- Model checkpointing
- Performance monitoring
- Deterministic training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multiOFF.git
cd multiOFF
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

1. **PrototypeGuidedPatchDecoder**: Extracts visual features using prototype-guided attention
2. **ContextGuidedTextDecoder**: Enhances text features using context-aware attention
3. **TextEnhancedCLIP**: Combines visual and textual features for classification

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended
- See requirements.txt for detailed dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{multiOFF2023,
  title={MultiOFF: A Multimodal Framework for Offensive Content Detection},
  author={Your Name},
  journal={Journal of Machine Learning},
  year={2023}
}
``` 