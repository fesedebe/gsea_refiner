import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from gsea_refiner.evaluation.split import get_train_test_blind_split
from gsea_refiner.preprocessing.clean import clean_gene_set_name

MAX_LENGTH = 48
FREEZE_UNTIL = 6
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
N_FOLDS = 5
SEED = 42
LR_CANDIDATES = [1e-5, 2e-5, 5e-5]

MODELS = {
    "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "scibert": "allenai/scibert_scivocab_uncased",
}

DEFAULT_MODELS = ["biomedbert"]
DEFAULT_OUTPUT_DIR = "data/models"


def prepare_data(df: pd.DataFrame):
    df = df[df["label"] != "Other"].copy()
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


class TrainingCurveLogger(TrainerCallback):
    def __init__(self):
        self.rows = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {"step": state.global_step, "epoch": state.epoch}
        row.update(logs)
        self.rows.append(row)

    def save(self, path):
        if self.rows:
            pd.DataFrame(self.rows).to_csv(path, index=False)


def _make_dataset(df_slice, tokenizer):
    ds = Dataset.from_pandas(
        df_slice[["pathway_clean", "label_id"]]
        .rename(columns={"label_id": "labels"})
        .reset_index(drop=True),
        preserve_index=False,
    )
    ds = ds.map(
        lambda ex: tokenizer(
            ex["pathway_clean"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        ),
        batched=True,
    )
    return ds


def _train_one(
    model_id,
    tokenizer,
    train_ds,
    val_ds,
    label2id,
    id2label,
    class_weights,
    lr,
    output_dir,
    curve_logger=None,
):
    num_labels = len(label2id)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
    except ValueError:
        model = BertForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
    freeze_bert_layers(model)

    WeightedTrainer = make_weighted_trainer_cls(class_weights)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    if curve_logger:
        callbacks.append(curve_logger)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
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
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    return trainer


def _load_tokenizer(model_id):
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except ValueError:
        return BertTokenizer.from_pretrained(model_id)


def _run_lr_sweep(model_name, model_id, df, label2id, id2label, class_weights, lr_candidates,
                  model_out_dir, seed):
    tokenizer = _load_tokenizer(model_id)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    lr_results = {}
    for lr in lr_candidates:
        print(f"\n  LR sweep: lr={lr}")

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(df["pathway_clean"], df["label_id"])
        ):
            print(f"    Fold {fold + 1}/{N_FOLDS}")

            train_ds = _make_dataset(df.iloc[train_idx], tokenizer)
            val_ds = _make_dataset(df.iloc[val_idx], tokenizer)

            fold_dir = os.path.join(model_out_dir, model_name, f"cv_lr_{lr}", f"fold_{fold}")
            curve_logger = TrainingCurveLogger()

            trainer = _train_one(
                model_id=model_id,
                tokenizer=tokenizer,
                train_ds=train_ds,
                val_ds=val_ds,
                label2id=label2id,
                id2label=id2label,
                class_weights=class_weights,
                lr=lr,
                output_dir=fold_dir,
                curve_logger=curve_logger,
            )

            val_results = trainer.evaluate()
            fold_f1 = val_results["eval_f1"]
            fold_scores.append(fold_f1)
            print(f"      fold {fold + 1}: macro F1 = {fold_f1:.4f}")

            curve_logger.save(os.path.join(fold_dir, "training_curve.csv"))

        mean_f1 = np.mean(fold_scores)
        std_f1 = np.std(fold_scores)
        lr_results[lr] = {"mean_f1": mean_f1, "std_f1": std_f1, "folds": fold_scores}
        print(f"    lr={lr}: {mean_f1:.4f} +/- {std_f1:.4f}")

    return tokenizer, lr_results


def _retrain_final(model_name, model_id, tokenizer, df, label2id, id2label, class_weights,
                   best_lr, model_out_dir, seed):
    print(f"\n  Retraining {model_name} on full training set with lr={best_lr}...")

    n_val = max(1, int(len(df) * 0.1))
    val_indices = df.sample(n=n_val, random_state=seed).index
    train_indices = df.index.difference(val_indices)

    final_train_ds = _make_dataset(df.iloc[train_indices], tokenizer)
    final_val_ds = _make_dataset(df.iloc[val_indices], tokenizer)

    final_dir = os.path.join(model_out_dir, model_name, "final")
    curve_logger = TrainingCurveLogger()

    trainer = _train_one(
        model_id=model_id,
        tokenizer=tokenizer,
        train_ds=final_train_ds,
        val_ds=final_val_ds,
        label2id=label2id,
        id2label=id2label,
        class_weights=class_weights,
        lr=best_lr,
        output_dir=final_dir,
        curve_logger=curve_logger,
    )

    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    curve_logger.save(os.path.join(final_dir, "training_curve.csv"))

    return final_dir


def fine_tune(
    model_names=None,
    model_out_dir=DEFAULT_OUTPUT_DIR,
    lr_candidates=None,
    seed=SEED,
):
    if model_names is None:
        model_names = DEFAULT_MODELS
    if lr_candidates is None:
        lr_candidates = LR_CANDIDATES

    train_df, _, _ = get_train_test_blind_split(seed=seed)
    df, label2id, id2label = prepare_data(train_df)
    num_labels = len(label2id)

    print(f"Training data: {len(df)} examples, {num_labels} classes (Other excluded)")
    for label, count in sorted(Counter(df["label"].tolist()).items()):
        print(f"  {label:25s}: {count}")

    class_weights = compute_class_weights(df["label_id"].tolist(), num_labels)

    all_results = {}

    for model_name in model_names:
        model_id = MODELS[model_name]
        print(f"\n{'=' * 60}")
        print(f"{model_name} ({model_id})")
        print(f"{'=' * 60}")

        tokenizer, lr_results = _run_lr_sweep(
            model_name=model_name,
            model_id=model_id,
            df=df,
            label2id=label2id,
            id2label=id2label,
            class_weights=class_weights,
            lr_candidates=lr_candidates,
            model_out_dir=model_out_dir,
            seed=seed,
        )

        best_lr = max(lr_results, key=lambda lr: lr_results[lr]["mean_f1"])
        print(f"\n  Best LR for {model_name}: {best_lr} "
              f"({lr_results[best_lr]['mean_f1']:.4f} +/- {lr_results[best_lr]['std_f1']:.4f})")

        final_dir = _retrain_final(
            model_name=model_name,
            model_id=model_id,
            tokenizer=tokenizer,
            df=df,
            label2id=label2id,
            id2label=id2label,
            class_weights=class_weights,
            best_lr=best_lr,
            model_out_dir=model_out_dir,
            seed=seed,
        )

        metadata = {
            "model_name": model_name,
            "model_id": model_id,
            "best_lr": best_lr,
            "num_labels": num_labels,
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
            "lr_sweep": {
                str(lr): {
                    "mean_f1": r["mean_f1"],
                    "std_f1": r["std_f1"],
                    "folds": r["folds"],
                }
                for lr, r in lr_results.items()
            },
            "config": {
                "max_length": MAX_LENGTH,
                "freeze_until": FREEZE_UNTIL,
                "num_epochs": NUM_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "warmup_ratio": WARMUP_RATIO,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "n_folds": N_FOLDS,
                "seed": seed,
            },
        }
        with open(os.path.join(final_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        all_results[model_name] = {
            "model_id": model_id,
            "best_lr": best_lr,
            "lr_results": lr_results,
            "final_dir": final_dir,
        }

        print(f"  Final model saved to {final_dir}")

    return all_results
