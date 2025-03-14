from transformers import pipeline
from typing import List, Dict

class PathwayClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        self.classifier = pipeline("zero-shot-classification", model=model_name)

    def classify_pathways(self, pathway_names: List[str], candidate_categories: List[str], batch_size=1, confidence_threshold=0.5) -> Dict[str, Dict[str, float]]:
        """Classifies pathways and assigns predicted categories with optional batching."""
        results = {}
        is_batching = batch_size > 1
        input_batches = [pathway_names[i:i + batch_size] for i in range(0, len(pathway_names), batch_size)] if is_batching else [pathway_names]

        for batch in input_batches:
            batch_results = self.classifier(batch, candidate_labels=candidate_categories)
            for j, pathway in enumerate(batch):
                scores = dict(zip(batch_results[j]["labels"], batch_results[j]["scores"]))
                best_category, best_score = max(scores.items(), key=lambda x: x[1])
                results[pathway] = {**scores, "Predicted Category": best_category if best_score >= confidence_threshold else "Other"}
        
        return results

