import torch
from transformers import pipeline
import pandas as pd
from typing import List, Dict, Union, Optional

class BinaryPathwayClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        self.classifier = pipeline("zero-shot-classification", model=model_name)

    def classify_pathways(
        self, pathway_names: List[str], categories: Union[str, List[str]], batch_size=1, 
        save_scores: bool = False, scores_output_file: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Classifies pathways into one or multiple categories."""
        
        results = {pathway: {} for pathway in pathway_names}
        scores_data = [] 

        if isinstance(categories, str):
            categories = [categories]

        for category in categories:
            print(f"🔹 Running classification for: {category}")
            is_batching = batch_size > 1
            input_batches = [pathway_names[i:i + batch_size] for i in range(0, len(pathway_names), batch_size)] if is_batching else [pathway_names]

            for batch in input_batches:
                batch_results = self.classifier(batch, candidate_labels=[category, f"Not {category}"])
                for j, pathway in enumerate(batch):
                    scores = dict(zip(batch_results[j]["labels"], batch_results[j]["scores"]))  # Map labels to scores
                    category_score = scores.get(category, 0.0)
                    not_category_score = scores.get(f"Not {category}", 0.0) 
                    results[pathway][category] = category_score if category_score >= not_category_score else -not_category_score
                    
                    if save_scores:
                        scores_data.append({"Pathway": pathway, "Category": category, "Category Score": category_score, "Not Category Score": not_category_score})

        if save_scores and scores_output_file:
            df = pd.DataFrame(scores_data)
            df.to_csv(scores_output_file, index=False)
            print(f"✅ Scores saved to {scores_output_file}")

        return results
    
    def get_predicted_categories(self, classification_results: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Determines the predicted category for each pathway based on the highest score."""
        predicted_categories = {}
        for pathway, category_scores in classification_results.items():
            best_category = max(category_scores, key=category_scores.get)
            predicted_categories[pathway] = best_category
        return predicted_categories