import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from gsea_refiner.preprocessing.clean import clean_gene_set_name


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def load_calibration(model_dir: str) -> dict:
    path = Path(model_dir) / "calibration.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def predict_categories(
    input_file: str,
    model_dir: str,
    output_file: str,
    confidence_threshold: Optional[float] = None,
    temperature: float = 1.0,
    scoring_method: str = "msp",
    max_length: int = 48,
    batch_size: int = 64,
):
    from gsea_refiner.classification.calibrate import energy_scores as _energy_scores

    df = pd.read_csv(input_file, sep=None, engine="python")
    if "pathway" not in df.columns:
        raise ValueError("Input file must contain a 'pathway' column.")

    df["cleaned"] = df["pathway"].apply(clean_gene_set_name)
    tokenizer, model = load_model(model_dir)

    has_other_class = "Other" in model.config.label2id

    if not has_other_class:
        cal = load_calibration(model_dir)
        if confidence_threshold is None and "threshold" in cal:
            confidence_threshold = cal["threshold"]
        if temperature == 1.0 and "temperature" in cal:
            temperature = cal["temperature"]
        if scoring_method == "msp" and "scoring_method" in cal:
            scoring_method = cal["scoring_method"]

    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch = df["cleaned"].iloc[i : i + batch_size].tolist()
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            outputs = model(**inputs)
            raw_logits = outputs.logits
            probs = softmax(raw_logits / temperature, dim=1)

            if not has_other_class and scoring_method == "energy":
                batch_logits = raw_logits.cpu().numpy()
                scores = _energy_scores(batch_logits, temperature)
            else:
                scores = probs.max(dim=1).values.cpu().numpy()

            for j in range(probs.shape[0]):
                pred_idx = torch.argmax(probs[j]).item()
                pred_label = model.config.id2label[pred_idx]
                confidence = float(scores[j])

                if not has_other_class and confidence_threshold is not None and confidence < confidence_threshold:
                    pred_label = "Other"

                all_labels.append(pred_label)
                all_confidences.append(confidence)

    df["predicted_category"] = all_labels
    df["confidence"] = all_confidences
    df[["pathway", "predicted_category", "confidence"]].to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
