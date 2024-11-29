import pandas as pd
import re

def extract_gene_set_names(file_path="data/input/gsea_results.csv") -> list:
    """
    Extract gene set (pathway) names from GSEA results file.
    
    Args:
        file_path (str): Path to the GSEA results file. Defaults to 'data/input/gsea_results.csv'.
        
    Returns:
        list: List of pathway names extracted from the file.
    """
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
    """
    Clean a single gene set name by removing prefixes and unwanted characters.
    
    Args:
        name (str): A single gene set (pathway) name.
        
    Returns:
        str: Cleaned pathway name.
    """
    # Convert to lowercase
    name = name.lower()
    
    # Remove everything before the first underscore, including the underscore
    name = re.sub(r'^[^_]*_', '', name)
    
    # Replace remaining underscores with spaces
    name = name.replace('_', ' ')
    
    # Strip leading and trailing spaces
    name = name.strip()
    
    return name

def process_gene_set_names(file_path="data/input/gsea_results.csv") -> list:
    """
    Extract and clean all gene set (pathway) names from GSEA results file.
    
    Args:
        file_path (str): Path to the GSEA results file. Defaults to 'data/input/gsea_results.csv'.
        
    Returns:
        list: List of cleaned pathway names.
    """
    # Extract pathway names
    pathways = extract_gene_set_names(file_path)
    
    # Clean each pathway name
    cleaned_pathways = [clean_gene_set_name(name) for name in pathways]
    return cleaned_pathways

if __name__ == "__main__":
    # Default GSEA file path
    input_file = "data/input/fGSEA_UCLAAllPatch_deseq_recur.txt"
    
    # Process gene set names
    cleaned_names = process_gene_set_names(input_file)
    
    # Print first 10 cleaned names for verification
    print("Cleaned Pathway Names (First 10):")
    print(cleaned_names[:10])
