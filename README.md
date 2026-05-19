# gsea_refiner

Supervised classifier that maps gene-set enrichment results to functional categories.

## What it does
GSEA and pathway enrichment analyses return thousands of significant gene sets with overlapping themes. While this helps identify individual pathways, it makes system-wide patterns harder to interpret. gsea_refiner fine-tunes biomedical transformer models to classify each pathway into a small set of functional categories. This helps identify dominant biological trends and enables cross-cohort comparison.

## Results
Evaluated on a gold-labeled test set (612 pathways, 8 classes):

| Method | Macro F1 | Accuracy |
|---|---|---|
| BiomedBERT 8-class | 0.7670 | 0.8775 |
| SciBERT 8-class | 0.7468 | 0.8497 |
| BioBERT 8-class | 0.7436 | 0.8448 |
| BERT-base 8-class | 0.6083 | 0.7647 |
| Regex baseline | 0.5523 | 0.7418 |
| BART-MNLI Zero-Shot | 0.2882 | 0.6373 |

## Quick Start

```bash
git clone https://github.com/fesedebe/gsea_refiner.git
cd /gsea_refiner
pip install -e .          # or: uv sync
pytest -q                 
python scripts/benchmark.py
```

## Package Structure
```
gsea_refiner/
  preprocessing/     clean, tokenize, filter pathway names
  labeling/          regex bulk labeling
  classification/    transformer training and inference
  enrichment/        rank-based KS enrichment (classifier-agnostic)
  evaluation/        evaluate models, baselines, metrics
  visualization/     evaluation plots and NES bar plots
  io.py              CSV/TSV reading utilities
```

## Documentation
- [docs/methods.md](docs/methods.md) -- evaluation methodology and training details