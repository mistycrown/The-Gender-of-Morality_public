# The Gender of Morality: A Cultural Geometry of the Imperial Chinese Lexicon

This repository contains the source code for the paper "The Gender of Morality".

## Project Structure

```
├── dictionaries/          # The constructed moral lexicon (verified by experts)
├── src/
│   ├── analysis/          # Core scripts for calculating gender bias (static & diachronic)
│   ├── axis/              # Scripts for constructing the high-dimensional gender axis
│   ├── preprocessing/     # LLM-assisted disambiguation pipeline
│   ├── validation/        # Statistical verification tests (Permutation Test)
│   ├── visualization/     # Scripts for generating figures
│   └── utils/             # Utility scripts for caching vectors
└── requirements.txt       # Python dependencies
```

## Requirements

- Python 3.8+
- PyTorch
- Vectors rely on `SIKU-BERT`.
- LLM annotation relies on DeepSeek API (optional, for reproduction of polysemy filtering).

## Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Core Analysis**:
   The main analysis is performed by calculating the projection of moral concepts onto the gender axis.
   ```bash
   python src/analysis/calculate_static_bias.py
   ```

## Note on Data

The raw corpus (Siku Quanshu & Xueheng Corpus) is not included due to copyright/size constraints. However, the core logic for processing vector embeddings is fully reproducible with any standard corpus format or by using SikuBERT directly.
