"""Transformer-enrichment pipeline: predict categories with fine-tuned model, then run KS enrichment per category."""

import argparse

import pandas as pd
import torch
from torch.nn.functional import softmax

from gsea_refiner.classification.predict import load_calibration, load_model
from gsea_refiner.enrichment.ks import run_category_enrichment
from gsea_refiner.preprocessing.clean import clean_gene_set_name


def predict_column(df, model_dir, max_length=48, batch_size=64):
    tokenizer, model = load_model(model_dir)
    cal = load_calibration(model_dir)
    threshold = cal.get("threshold")
    temperature = cal.get("temperature", 1.0)

    cleaned = df["pathway"].apply(clean_gene_set_name).tolist()
    labels = []

    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding=True, max_length=max_length,
            )
            logits = model(**inputs).logits / temperature
            probs = softmax(logits, dim=1)
            for j in range(probs.shape[0]):
                idx = torch.argmax(probs[j]).item()
                label = model.config.id2label[idx]
                conf = probs[j, idx].item()
                if threshold is not None and conf < threshold:
                    label = "Other"
                labels.append(label)

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Transformer predict → KS enrichment (parallel to gsea_squared.py)"
    )
    parser.add_argument("--input", required=True, help="GSEA results file (CSV/TSV with pathway + NES columns)")
    parser.add_argument("--model-dir", default="data/models/biomedbert/final")
    parser.add_argument("--catmap", default="data/config/category_keywords.csv")
    parser.add_argument("--savename", default="data/output/transformer_enrichment")
    args = parser.parse_args()

    categories = pd.read_csv(args.catmap)["Category"].tolist()

    sep = "\t" if args.input.endswith(".txt") else None
    df = pd.read_csv(args.input, sep=sep, engine="python")

    print(f"Predicting categories with model: {args.model_dir}")
    df["Category"] = predict_column(df, args.model_dir)

    print("\nCategory distribution:")
    for cat, count in df["Category"].value_counts().items():
        print(f"  {cat:25s}: {count}")

    results = run_category_enrichment(
        df,
        categories=categories,
        prediction_col="Category",
        savename=args.savename,
        verbose=True,
    )

    print("\nEnrichment results:")
    print(results["categories"].to_string(index=False))


if __name__ == "__main__":
    main()
