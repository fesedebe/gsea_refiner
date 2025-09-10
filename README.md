# gsea_refiner
 A Python tool that summarizes long lists of enriched pathways into functional categories.

## Context
 GSEA/pathway enrichment analysis often returns thousands of significant gene sets with overlapping themes. While this helps identify individual pathways, it makes system-wide patterns harder to interpret. GSEA-refiner fine-tunes a biomedical transformer model (BioBERT) to map enriched pathways into a small set of interpretable themes (e.g., Cell-cycle, Immune, Neuro). This helps identify dominant trends quickly and compare signals across cohorts or treatment stages.
 
## Features
- Takes GSEA results and predicts per-term biological categories based on semantic content
- Partially fine-tuned model outperformed traditional (TF-IDF, BiLSTM) and general-domain (DistilBERT) models
- Ranks top genes within categories and generates visualizations summarizing enrichment trends

## Workflow
![GSEA-refiner Workflow](docs/gsea_refiner_workflow.png)

## Documentation
See [methods.md](docs/methods.md) for technical and evaluation details.

## Installation
Clone the repository and install:

### For development (editable mode)
```bash
git clone https://github.com/fesedebe/gsea_refiner.git
cd gsea_refiner
pip install -e .
```

## Dependencies
See `requirements.txt` for full list.
- `transformers`
- `datasets`
- `scikit-learn`
- `pandas`
- `torch`

## Repo Layout
```
gsea_refiner/categorization/   # Rule-based labeling, transformer classification
gsea_refiner/preprocessing/    # Token filtering, weighting
gsea_refiner/visualization/    # Plotting utilities
scripts # Workflows
```

## Status  
Developed for a UCLA Bioinformatics PhD dissertation on adaptive therapy resistance in aggressive cancers, GSEA-refiner has been used to compare enrichment trends across cohorts. It is actively maintained and supports both exploratory and publication-ready analyses.