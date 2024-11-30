import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_corpus(corpus_path: str) -> list:
    """
    Load the tokenized corpus from a text file.
    
    Args:
        corpus_path (str): Path to the corpus file.
        
    Returns:
        list: A list of strings, where each string represents a pathway's tokenized words.
    """
    with open(corpus_path, "r") as f:
        corpus = [line.strip() for line in f.readlines()]
    return corpus

def compute_tfidf(corpus: list, max_features: int = 1000) -> pd.DataFrame:
    """
    Compute TF-IDF scores for terms in the corpus.
    
    Args:
        corpus (list): List of tokenized pathway strings.
        max_features (int): Maximum number of terms to consider based on importance.
        
    Returns:
        pd.DataFrame: DataFrame containing terms and their TF-IDF scores.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1  # Sum TF-IDF scores across all pathways
    return pd.DataFrame({"Term": terms, "TF-IDF": scores}).sort_values(by="TF-IDF", ascending=False)

def save_tfidf_scores(tfidf_df: pd.DataFrame, output_path: str) -> None:
    """
    Save TF-IDF scores to a CSV file.
    
    Args:
        tfidf_df (pd.DataFrame): DataFrame containing terms and their TF-IDF scores.
        output_path (str): Path to save the TF-IDF scores as a CSV file.
    """
    tfidf_df.to_csv(output_path, index=False)
    print(f"TF-IDF scores saved to {output_path}")

if __name__ == "__main__":
    # File paths
    corpus_path = "data/intermediate/corpus.txt"
    output_path = "data/intermediate/tfidf_scores.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #Load the tokenized corpus
    print("[Step 1] Loading tokenized corpus...")
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} pathways from {corpus_path}.")

    # Compute TF-IDF scores
    print("[Step 2] Computing TF-IDF scores...")
    tfidf_df = compute_tfidf(corpus)
    print(f"Computed TF-IDF scores for {len(tfidf_df)} terms.")

    # Save TF-IDF scores
    print("[Step 3] Saving TF-IDF scores...")
    save_tfidf_scores(tfidf_df, output_path)

    print("TF-IDF analysis completed successfully.")
    #call: python3 -m scripts.analysis.gsea_refiner.tfidf_analysis