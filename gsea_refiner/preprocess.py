import pandas as pd
import re
import json
import os
from typing import List, Optional, Set

#1. Extract & Clean Gene Set Names
def extract_gene_set_names(file_path: str) -> List[str]:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".txt"):
        df = pd.read_table(file_path)
    else:
        raise ValueError("Unsupported file format. Only '.csv' and '.txt' are supported.")
    
    if 'pathway' not in df.columns:
        raise ValueError("Input file must contain a 'pathway' column.")
    
    return df['pathway'].tolist()

def clean_gene_set_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'^[^_]*_', '', name)
    name = name.replace('_', ' ').strip()
    return name

def process_gene_set_names(file_path: str) -> List[str]:
    pathways = extract_gene_set_names(file_path)
    return [clean_gene_set_name(name) for name in pathways]

#2. Tokenization
STOPWORDS = {"and", "of", "the", "in", "for", "with", "on", "to", "a"}
def tokenize_name(name: str, stopwords: Optional[Set[str]] = STOPWORDS) -> List[str]:
    tokens = re.split(r'\s+', name)
    if stopwords is not None:
        print("removing stopwords")
        tokens = [word for word in tokens if word not in stopwords]
    return tokens

def tokenize_corpus(names: List[str], stopwords: Optional[Set[str]] = STOPWORDS) -> List[List[str]]:
    return [tokenize_name(name, stopwords) for name in names]

#3. Saving Tokenized Corpus
def save_corpus_to_txt(tokenized_names: List[List[str]], output_path: str) -> None:
    with open(output_path, "w") as f:
        for tokens in tokenized_names:
            f.write(" ".join(tokens) + "\n")

def save_corpus_to_json(tokenized_names: List[List[str]], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(tokenized_names, f, indent=2)