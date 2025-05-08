import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
import os


def load_dataset_from_csv(path: str) -> Dataset:
    df = pd.read_csv(path)
    if 'label' not in df.columns or 'pathway' not in df.columns:
        raise ValueError("CSV must contain 'pathway' and 'label' columns")

    label2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    df['label_id'] = df['label'].map(label2id)
    dataset = Dataset.from_pandas(df[['pathway', 'label_id']])
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset, label2id


def tokenize_function(example, tokenizer):
    return tokenizer(example["pathway"], truncation=True, padding=False)


def fine_tune_biobert(
    input_csv="data/training/labeled_pathways.csv",
    model_out_dir="data/models/biobert_finetuned",
    model_name="dmis-lab/biobert-base-cased-v1.1",
    num_train_epochs=5,
    batch_size=16,
    lr=2e-5
):
    dataset, label2id = load_dataset_from_csv(input_csv)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label={v: k for k, v in label2id.items()},
        label2id=label2id
    )

    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=model_out_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{model_out_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(model_out_dir)
    tokenizer.save_pretrained(model_out_dir)

    print(f"Model saved to {model_out_dir}")
    return model_out_dir
