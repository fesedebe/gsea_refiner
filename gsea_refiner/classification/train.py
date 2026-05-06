import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
import os
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import numpy as np


def load_dataset_from_csv(path: str):
    df = pd.read_csv(path)
    if 'label' not in df.columns or 'pathway' not in df.columns:
        raise ValueError("CSV must contain 'pathway' and 'label' columns")

    label2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    df['label_id'] = df['label'].map(label2id)
    return df, label2id


def tokenize_function(example, tokenizer):
    return tokenizer(example["pathway"], truncation=True, padding=False)


def compute_class_weights(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def freeze_bert_layers(model, freeze_until=8):
    for name, param in model.bert.named_parameters():
        if "encoder.layer." in name:
            layer_num = int(name.split(".")[2])
            if layer_num < freeze_until:
                param.requires_grad = False


def fine_tune_biobert(
    input_csv="data/training/labeled_pathways.csv",
    model_out_dir="data/models/biobert_finetuned",
    model_name="dmis-lab/biobert-base-cased-v1.1",
    num_train_epochs=5,
    batch_size=16,
    lr=2e-5
):
    df, label2id = load_dataset_from_csv(input_csv)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    class_weights = compute_class_weights(df['label_id'].tolist())

    def compute_loss(model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['pathway'], df['label_id'])):
        print(f"\nTraining fold {fold+1}/5")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_dataset = Dataset.from_pandas(train_df[['pathway', 'label_id']])
        val_dataset = Dataset.from_pandas(val_df[['pathway', 'label_id']])

        train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label={v: k for k, v in label2id.items()},
            label2id=label2id
        )

        freeze_bert_layers(model)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=os.path.join(model_out_dir, f"fold_{fold}"),
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=os.path.join(model_out_dir, f"fold_{fold}/logs"),
            logging_steps=10,
            load_best_model_at_end=True,
            save_total_limit=1,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_loss=compute_loss
        )

        trainer.train()
        model.save_pretrained(os.path.join(model_out_dir, f"fold_{fold}"))
        tokenizer.save_pretrained(os.path.join(model_out_dir, f"fold_{fold}"))

    print(f"\n Completed 5-fold training. Models saved to {model_out_dir}")
