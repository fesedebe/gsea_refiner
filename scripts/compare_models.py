"""Phase 4 Task 2: Model comparison experiment.

Runs BiomedBERT/PubMedBERT, BioBERT v1.1, and optionally SciBERT through identical
5-fold stratified CV on the 7-class training data (Other excluded).
Reports macro F1 per fold and mean +/- std for model selection.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --models biomedbert biobert scibert
    python scripts/compare_models.py --lr 1e-5 2e-5 5e-5  # LR sweep

Output:
    data/output/model_comparison.csv
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from gsea_refiner.evaluation.split import get_train_test_blind_split
from gsea_refiner.preprocessing.clean import clean_gene_set_name

MODELS = {
    "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "scibert": "allenai/scibert_scivocab_uncased",
}

MAX_LENGTH = 48
FREEZE_UNTIL = 6
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
N_FOLDS = 5
SEED = 42


def prepare_data(train_df: pd.DataFrame):
    df = train_df[train_df["label"] != "Other"].copy()
    df["pathway_clean"] = df["pathway"].apply(clean_gene_set_name)

    labels_sorted = sorted(df["label"].unique())
    label2id = {label: idx for idx, label in enumerate(labels_sorted)}
    id2label = {v: k for k, v in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    return df.reset_index(drop=True), label2id, id2label


def compute_class_weights(labels, num_classes):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def freeze_bert_layers(model, freeze_until=FREEZE_UNTIL):
    for attr in ("bert", "roberta"):
        if hasattr(model, attr):
            encoder = getattr(model, attr)
            break
    else:
        print("  Warning: could not find encoder — skipping layer freeze")
        return

    for name, param in encoder.named_parameters():
        if "encoder.layer." in name:
            layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_num < freeze_until:
                param.requires_grad = False


def make_weighted_trainer_cls(class_weights):
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(logits.device)
            )
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"eval_f1": f1_score(labels, preds, average="macro", zero_division=0)}


def run_cv_for_model(model_name, model_id, df, label2id, id2label, lr, output_dir):
    num_labels = len(label2id)
    class_weights = compute_class_weights(df["label_id"].tolist(), num_labels)
    WeightedTrainer = make_weighted_trainer_cls(class_weights)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize_fn(examples):
        return tokenizer(
            examples["pathway_clean"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df["pathway_clean"], df["label_id"])
    ):
        print(f"  {model_name} lr={lr} — fold {fold + 1}/{N_FOLDS}")

        train_data = Dataset.from_pandas(
            df.iloc[train_idx][["pathway_clean", "label_id"]]
            .rename(columns={"label_id": "labels"})
            .reset_index(drop=True),
            preserve_index=False,
        )
        val_data = Dataset.from_pandas(
            df.iloc[val_idx][["pathway_clean", "label_id"]]
            .rename(columns={"label_id": "labels"})
            .reset_index(drop=True),
            preserve_index=False,
        )

        train_data = train_data.map(tokenize_fn, batched=True)
        val_data = val_data.map(tokenize_fn, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        freeze_bert_layers(model)

        fold_dir = os.path.join(output_dir, model_name, f"lr_{lr}", f"fold_{fold}")

        training_args = TrainingArguments(
            output_dir=fold_dir,
            learning_rate=lr,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=os.path.join(fold_dir, "logs"),
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=1,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            report_to="none",
            seed=SEED,
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
            ],
        )

        trainer.train()

        val_results = trainer.evaluate()
        fold_f1 = val_results["eval_f1"]
        fold_scores.append(fold_f1)
        print(f"    fold {fold + 1}: macro F1 = {fold_f1:.4f}")

    return fold_scores


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 Task 2: Model comparison experiment"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["biomedbert", "biobert"],
        choices=list(MODELS.keys()),
    )
    parser.add_argument(
        "--lr",
        nargs="+",
        type=float,
        default=[2e-5],
        help="Learning rates to try (default: 2e-5; sweep: 1e-5 2e-5 5e-5)",
    )
    parser.add_argument("--output", default="data/output/model_comparison.csv")
    parser.add_argument("--output-dir", default="data/models/comparison")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("Loading data...")
    train_df, _, _ = get_train_test_blind_split(seed=args.seed)
    df, label2id, id2label = prepare_data(train_df)

    print(f"Training data: {len(df)} examples, {len(label2id)} classes")
    for label, count in sorted(Counter(df["label"].tolist()).items()):
        print(f"  {label:25s}: {count}")
    print()

    results = []

    for model_name in args.models:
        model_id = MODELS[model_name]
        for lr in args.lr:
            print(f"\n{'=' * 60}")
            print(f"{model_name} ({model_id}) — lr={lr}")
            print(f"{'=' * 60}")

            fold_scores = run_cv_for_model(
                model_name=model_name,
                model_id=model_id,
                df=df,
                label2id=label2id,
                id2label=id2label,
                lr=lr,
                output_dir=args.output_dir,
            )

            results.append(
                {
                    "model": model_name,
                    "model_id": model_id,
                    "lr": lr,
                    "mean_f1": np.mean(fold_scores),
                    "std_f1": np.std(fold_scores),
                    **{f"fold_{i}_f1": s for i, s in enumerate(fold_scores)},
                }
            )

            print(
                f"\n  {model_name} lr={lr}: "
                f"{np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}"
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['model']:12s} lr={r['lr']:.0e}: {r['mean_f1']:.4f} +/- {r['std_f1']:.4f}")

    winner = max(results, key=lambda r: r["mean_f1"])
    print(
        f"\nWinner: {winner['model']} lr={winner['lr']:.0e} "
        f"({winner['mean_f1']:.4f} +/- {winner['std_f1']:.4f})"
    )
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
