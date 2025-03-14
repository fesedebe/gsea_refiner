import pandas as pd

def read_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".txt"):
        return pd.read_table(file_path)
    else:
        raise ValueError("Unsupported file format. Only '.csv' and '.txt' are supported.")
    
def load_corpus(corpus_file: str) -> list:
    with open(corpus_file, "r") as file:
        return [line.strip() for line in file.readlines() if line.strip()]