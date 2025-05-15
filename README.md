# gsea_refiner
 A Python tool for categorizing and interpreting pathway enrichment results using transformer models and keyword-based labeling.

## Overview
 Pathway enrichment analysis often produces thousands of significant gene sets with overlapping biological themes, making it difficult to extract meaningful insights. GSEA-refiner helps organize these results into biologically interpretable categories using keyword matching and fine-tuned BioBERT classification. The tool supports modular preprocessing, model training, prediction, and visualization across transcriptomics datasets.

## Installation
Clone the repository and install:

### For development (editable mode)
```bash
git clone https://github.com/fesedebe/gsea_refiner.git
cd gsea_refiner
pip install -e .
