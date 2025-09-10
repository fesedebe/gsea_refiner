# Methods, Data, and Evaluation

## Preprocessing
Tokenize pathway names from input data: e.g: ”GOBP_IMMUNE_RESPONSE_PATHWAY" → [immune response pathway]

## Goals
- **Input**: Gene set names (e.g., `"interferon gamma signaling"`)
- **Task**: Multi-class classification
- **Output**: High-level functional themes like `"immune"` or `"cell cycle"`

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
---

## Model Training
Next, BioBERT was fine-tuned for pathway name classification.

- **Base model**: `dmis-lab/biobert-base-cased-v1.1`
- **Training strategy**:
  - Freeze lower 8 layers, train top 4
  - Cross-entropy loss with class weighting
  - Tokenizer: BioBERT default

- **Evaluation**:
  - 5-fold stratified cross-validation
  - Final model: Partial fine-tuned BioBERT (Macro-F1: 0.81)

---

## Benchmarking
| Method                         | Macro-F1 |
|--------------------------------|----------|
| TF-IDF + Logistic Regression   | 0.45    |
| GloVe + BiLSTM                 | 0.65    |
| DistilBERT                     | 0.75    |
| BioBERT (Full fine tuning)     | 0.77    |

---

## Future Directions
- Benchmark other biomedical LLMs (SciBERT, PubMedBERT) or multi-omics embeddings to improve classification.
- Use generative models (GPT) to propose new category groupings beyond training data.
- Incorporate retrieval-augmented classification to pull descriptions from external pathway databases.