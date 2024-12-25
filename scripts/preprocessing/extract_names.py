import pandas as pd
import re

def extract_gene_set_names(file_path="data/input/gsea_results.csv") -> list:
    # Load GSEA results into a DataFrame
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".txt"):
        df = pd.read_table(file_path)
    else:
        raise ValueError("Unsupported file format. Only '.csv' and '.txt' are supported.")
    
    # Check if 'pathway' column exists
    if 'pathway' not in df.columns:
        raise ValueError("Input file must contain a 'pathway' column.")
    
    # Extract pathway names and convert to a list
    pathways = df['pathway'].tolist()
    return pathways

def clean_gene_set_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'^[^_]*_', '', name) # Remove everything before & including the first underscore
    name = name.replace('_', ' ')
    name = name.strip()
    
    return name

def process_gene_set_names(file_path="data/input/gsea_results.csv") -> list:
    # Extract pathway names
    pathways = extract_gene_set_names(file_path)
    
    # Clean each pathway name
    cleaned_pathways = [clean_gene_set_name(name) for name in pathways]
    return cleaned_pathways

if __name__ == "__main__":
    input_file = "data/input/gsea_results.csv"
    cleaned_names = process_gene_set_names(input_file)
    
    # Print first 10 cleaned names for verification
    print("Cleaned Pathway Names (First 10):")
    print(cleaned_names[:10])
