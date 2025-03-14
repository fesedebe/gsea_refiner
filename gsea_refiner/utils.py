import pandas as pd

def read_file(file_path: str) -> pd.DataFrame:
    """reads csv or txt file info pandas df"""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".txt"):
        return pd.read_table(file_path)
    else:
        raise ValueError("Unsupported file format. Only '.csv' and '.txt' are supported.")