# Methods, Data, and Evaluation

## Task

- **Input**: Gene set names from GSEA results (e.g., `"GOBP_IMMUNE_RESPONSE"`)
- **Task**: Multi-class classification into 8 categories
- **Output**: Functional theme (Cell Cycle, Differentiation, Epigenetic Reg., Immune, Lipid, Neuro, DNA Repair, or Other)

| Example Pathway | Predicted Category |
|---|---|
| GOBP_INTERFERON_GAMMA_SIGNALING | Immune |
| REACTOME_MITOTIC_SPINDLE_CHECKPOINT | Cell Cycle |
| GOBP_RESPONSE_TO_OXIDATIVE_STRESS | DNA Repair |

## Preprocessing

Pathway names are cleaned and tokenized: strip database prefix, replace underscores with spaces, lowercase. Example: `"GOBP_IMMUNE_RESPONSE_PATHWAY"` becomes `"immune response pathway"`.

## Labeled Training Data

7,084 pathway-category pairs curated via a hybrid process: regex-seeded labeling (adapted from [Balanis, Sheu, Esedebe et al., 2019](https://doi.org/10.1016/j.ccell.2019.06.005)), followed by manual curation.

Split sizes: Training n=5,507 / Test n=612 (90/10 stratified, seed=42).

## Model Training

Four transformer models fine-tuned with "Other" as an explicit 8th class:

- **BioBERT** (`dmis-lab/biobert-base-cased-v1.1`)
- **BiomedBERT** (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`)
- **SciBERT** (`allenai/scibert_scivocab_uncased`)
- **BERT-base** (`google-bert/bert-base-uncased`)

Training recipe: freeze bottom 6 layers, fine-tune top 6 + classification head. 5-fold stratified CV with LR sweep [1e-5, 2e-5, 5e-5]. Class-weighted CrossEntropyLoss. 20 epochs max with early stopping (patience=3) on validation F1. Best config retrained on full training set for final model.

## Evaluation

| Method | Macro F1 | Macro F1 (excl. Other) | Accuracy |
|---|---|---|---|
| BiomedBERT 8-class | 0.7670 | 0.7449 | 0.8775 |
| SciBERT 8-class | 0.7468 | 0.7252 | 0.8497 |
| BioBERT 8-class | 0.7436 | 0.7209 | 0.8448 |
| BERT-base 8-class | 0.6083 | 0.5739 | 0.7647 |
| Regex baseline | 0.5523 | 0.5124 | 0.7418 |
| BART-MNLI Zero-Shot | 0.2882 | 0.2196 | 0.6373 |

**Notes:**
- Regex comparison is circular (regex generated the training labels). Fair comparison is between learned methods.
- "Excl. Other" macro F1 evaluates only the 7 biological categories, removing the dominant Other class (69% of test set).
- BART-MNLI is a zero-shot baseline using `facebook/bart-large-mnli` with no fine-tuning.

## Future Directions

- Multi-label classification for cross-category pathways (e.g., "immune cell differentiation")
- Collection-type prefix tokens ([BP], [MF], [CC]) to disambiguate GO molecular function terms
- MSigDB description augmentation for richer training signal
