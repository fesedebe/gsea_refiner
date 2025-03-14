import os
import pandas as pd
from gsea_refiner.classification.zsl_classifier import PathwayClassifier
from gsea_refiner.utils import load_corpus
from typing import List

def run_classification_pipeline(corpus_file: str, output_file: str, categories: List[str], batch_size=1, confidence_threshold=0.5):
    """Runs ZSL classification (in batches) and adds predicted category."""
    pathway_names = load_corpus(corpus_file)
    print(f"Loaded {len(pathway_names)} pathways from {corpus_file}")

    classification_results = PathwayClassifier().classify_pathways(pathway_names, categories, batch_size, confidence_threshold)

    results_df = pd.DataFrame([
        {"Pathway": pathway, **classification_results[pathway]}
        for pathway in classification_results
    ])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"Classification results saved to {output_file}")

if __name__ == "__main__":
    corpus_file = "data/intermediate/corpus_filtered.txt"  
    output_file = "data/output/classification_results.csv"
    categories = ["DNA Repair", "Cell Cycle Regulation"]

    run_classification_pipeline(corpus_file, output_file, categories, batch_size=100, confidence_threshold=0.5)
