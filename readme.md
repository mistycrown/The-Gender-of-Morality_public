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


