import os
import pandas as pd
from gsea_refiner.classification.zsl_classifier import BinaryPathwayClassifier
from gsea_refiner.utils import load_corpus
from typing import List, Union, Optional

def run_classification_pipeline(
    corpus_file: str, output_file: str, categories: Union[str, List[str]], batch_size=1, 
    save_scores: bool = False, scores_output_file: Optional[str] = None
):
    """Runs binary classification for one or multiple categories and selects the strongest match."""
    
    pathway_names = load_corpus(corpus_file)
    print(f"✅ Loaded {len(pathway_names)} pathways from {corpus_file}")

    classifier = BinaryPathwayClassifier()
    classification_results = classifier.classify_pathways(
        pathway_names, categories, batch_size, save_scores, scores_output_file
    )
    predicted_categories = classifier.get_predicted_categories(classification_results)

    results_list = []
    for pathway, category_scores in classification_results.items():
        row = {"Pathway": pathway, **category_scores, "Predicted Category": predicted_categories[pathway]}
        results_list.append(row)

    results_df = pd.DataFrame(results_list)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"✅ Classification results saved to {output_file}")

if __name__ == "__main__":
    corpus_file = "data/intermediate/corpus_filtered.txt"  
    output_file = "data/output/classification_results.csv"
    scores_output_file = "data/output/classification_scores.csv" 
    categories = ["DNA Repair", "Cell Cycle"]  

    run_classification_pipeline(corpus_file, output_file, categories, batch_size=100, save_scores=True, scores_output_file=scores_output_file)