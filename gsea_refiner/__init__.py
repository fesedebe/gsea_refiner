"""gsea_refiner: a supervised classification layer for gene-set → functional category.

Regex bulk-labels pathways cheaply; manual review catches the misses; the curated
set trains a transformer (BioBERT/PubMedBERT) that generalizes to unseen pathways.
The same predicted categories then feed a rank-based KS enrichment step, so the
whole pipeline can be applied to any GSEA output.

Pipeline:

    preprocessing  →  labeling   →  (manual review)  →  classification  →  enrichment
       Step 1         Step 2          Step 3              Steps 4-5         Step 6
    clean+tokenize   regex bulk-     gold set in       train+predict     KS per category
                       label         scripts/          BioBERT           on NES rank

evaluation/ benchmarks classifiers (regex, TF-IDF, BioBERT) against the gold set;
visualization/ plots the per-category NES results.
"""
