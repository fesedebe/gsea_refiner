from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from gsea_refiner.labeling.regex import label_pathways_by_regex
from gsea_refiner.preprocessing.clean import clean_gene_set_name

DEFAULT_KEYWORDS_PATH = Path("data/config/category_keywords.csv")

Predictor = Callable[[List[str]], List[str]]


def make_regex_predictor(keywords_path: Path = DEFAULT_KEYWORDS_PATH) -> Predictor:
    """Predicts category by regex match on pathway name. Returns 'Other' if no pattern matches.

    The bundled regex patterns target MSigDB-style names (CELL_CYCLE, _COA_, etc).
    Pathway names in the gold set are pre-cleaned (spaces, lowercase), so we
    re-introduce underscores before matching so the existing patterns still apply.
    """
    cats = pd.read_csv(keywords_path)
    categories = cats["Category"].tolist()
    cat_terms = cats["Regex"].tolist()

    def predict(pathways: List[str]) -> List[str]:
        df = pd.DataFrame({"pathway": [p.replace(" ", "_") for p in pathways]})
        labeled = label_pathways_by_regex(
            df, categories, cat_terms, col="pathway", label_col="label"
        )
        return labeled["label"].tolist()

    return predict


def make_tfidf_logreg_predictor(
    train_df: pd.DataFrame,
    pathway_col: str = "pathway",
    label_col: str = "label",
    seed: int = 42,
) -> Predictor:
    """TF-IDF (word 1- and 2-grams) + class-weighted multinomial LogReg.

    Pathway names are cleaned (`clean_gene_set_name`) before vectorization so
    the word tokenizer can split MSigDB-style names like
    `HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION` into useful tokens.
    """
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, random_state=seed, class_weight="balanced"
                ),
            ),
        ]
    )
    X_train = [clean_gene_set_name(p) for p in train_df[pathway_col].tolist()]
    pipeline.fit(X_train, train_df[label_col].tolist())

    def predict(pathways: List[str]) -> List[str]:
        X = [clean_gene_set_name(p) for p in pathways]
        return pipeline.predict(X).tolist()

    return predict


def make_transformer_predictor(
    model_dir: str,
    confidence_threshold: Optional[float] = None,
    temperature: float = 1.0,
    max_length: int = 48,
    batch_size: int = 64,
) -> Predictor:
    import json

    import torch
    from torch.nn.functional import softmax
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    cal_path = Path(model_dir) / "calibration.json"
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        if confidence_threshold is None and "threshold" in cal:
            confidence_threshold = cal["threshold"]
        if temperature == 1.0 and "temperature" in cal:
            temperature = cal["temperature"]

    def predict(pathways: List[str]) -> List[str]:
        cleaned = [clean_gene_set_name(p) for p in pathways]
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
                    if confidence_threshold is not None and conf < confidence_threshold:
                        label = "Other"
                    labels.append(label)
        return labels

    return predict
