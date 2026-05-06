import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from gsea_refiner.preprocessing.clean import clean_gene_set_name


def load_model(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict_categories(input_file, model_dir, output_file):
    df = pd.read_csv(input_file, sep=None, engine="python")
    if "pathway" not in df.columns:
        raise ValueError("Input file must contain a 'pathway' column.")

    df["cleaned"] = df["pathway"].apply(clean_gene_set_name)

    tokenizer, model = load_model(model_dir)

    predictions = []
    with torch.no_grad():
        for name in df["cleaned"]:
            inputs = tokenizer(name, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1).squeeze()
            pred_idx = torch.argmax(probs).item()
            pred_label = model.config.id2label[pred_idx]
            confidence = probs[pred_idx].item()
            predictions.append((pred_label, confidence))

    df["predicted_category"] = [p[0] for p in predictions]
    df["confidence"] = [p[1] for p in predictions]
    df[["pathway", "predicted_category", "confidence"]].to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")