# GSEA-refiner Documentation

## Problem Framing

GSEA-refiner is a Python/transformer-based tool for classifying enriched gene sets into higher-level biological categories (e.g., “immune,” “repair”). It reduces redundancy and improves interpretability of GSEA outputs across transcriptomic (bulk, single-cell, spatial) comparisons.

- **Input**: Gene set names (e.g., `"interferon gamma signaling"`)
- **Task**: Multi-class classification
- **Output**: High-level functional labels like `"immune"` or `"cell cycle"`

| Example Token                  | Predicted Category |
|-------------------------------|---------------------|
| interferon gamma signaling    | Immune              |
| mitotic spindle checkpoint    | Cell Cycle          |
| response to oxidative stress  | Stress Response     |

---

## Labeled Training Data

Pathway–category label pairs were curated using keyword-based rules. This was adapted from gsea-squared method described in co-authored publication [Balanis, Sheu, Esedebe et al., 2019](https://doi.org/10.1016/j.ccell.2019.06.005), where pathway keywords were grouped into biological themes for SCN profiling.

- **Sources**: Pathway names from tools like GSEA (e.g., `"REACTOME_DNA_REPAIR"`)
- **Logic**:
  - `"immune"` → `IMMUN|CYTOKINE|MHC`
  - `"repair"` → `REPAIR|FANCONI|DAMAGE`
  - `"cycle"` → `CELL_CYCLE|MITOSIS|DIVISION`

These labels were used to supervise initial model training.

```python
("REACTOME_INTERFERON_SIGNALING", "immune")
("REACTOME_HOMOLOGOUS_RECOMBINATION", "repair")
```

---

## Model Training (BioBERT Fine-Tuning)

Next, BioBERT was fine-tuned for pathway name classification.

- **Base model**: `dmis-lab/biobert-base-cased-v1.1`
- **Training strategy**:
  - Freeze lower 8 layers, train top 4
  - Cross-entropy loss with class weighting
  - Tokenizer: BioBERT default

- **Evaluation**:
  - 5-fold stratified cross-validation
  - Macro-F1: ~0.80, Accuracy: ~0.82

---

## Baseline Comparison

| Method                         | Macro-F1 |
|--------------------------------|----------|
| TF-IDF + Logistic Regression   | ~0.68    |
| GloVe + K-means                | Noisy    |

---

## Future Work

- Try contrastive loss or RAG-based summarization
- Build a web UI for fast enrichment interpretation

---

## File Structure

```
categorization/   # Rule-based labeling, transformer classification
preprocessing/    # Token filtering, weighting
visualization/    # Plotting utilities
```