import os
import pandas as pd
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ks_2samp


# Load the corpus
def load_corpus(corpus_path: str) -> list:
    with open(corpus_path, "r") as f:
        return [line.strip() for line in f.readlines()]


# Compute TF-IDF
def compute_tfidf(corpus: list, max_features: int = 1000, stop_words: list = None) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    return pd.DataFrame({"Term": terms, "TF-IDF": scores}).sort_values(by="TF-IDF", ascending=False)


# Run KS test for significance
def run_significance_tests(tfidf_terms: pd.DataFrame, gsea_results: pd.DataFrame, pval_threshold: float = 0.05) -> pd.DataFrame:
    results = []

    for term in tfidf_terms["Term"]:
        # Subset pathways containing the term
        test_pathways = gsea_results[gsea_results["pathway"].str.contains(term, case=False)]
        background_pathways = gsea_results[~gsea_results["pathway"].str.contains(term, case=False)]

        # Skip if there are too few pathways
        if len(test_pathways) < 2 or len(background_pathways) < 2:
            continue
        #pdb.set_trace()
        # Perform KS test on NES scores
        p_value = ks_2samp(test_pathways.index, background_pathways.index).pvalue

        # Store results
        results.append({
            "Term": term,
            "TF-IDF": tfidf_terms.loc[tfidf_terms["Term"] == term, "TF-IDF"].values[0],
            "KS_p-value": p_value,
            "Significant": p_value <= pval_threshold
        })

    return pd.DataFrame(results)


# Save results
def save_results(output_path: str, df: pd.DataFrame) -> None:
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # File paths
    corpus_path = "data/intermediate/corpus.txt"
    gsea_results_path = "data/input/fGSEA_UCLAAllPatch_deseq_recur.txt"
    tfidf_output_path = "data/intermediate/tfidf_scores.csv"
    significance_output_path = "data/intermediate/significant_terms.csv"

    # Stopwords for filtering
    stop_words = ["process", "pathway", "regulation", "response", "activity", "positive", "negative"]

    # Step 1: Load corpus
    print("[Step 1] Loading tokenized corpus...")
    corpus = load_corpus(corpus_path)

    # Step 2: Compute TF-IDF scores
    print("[Step 2] Computing TF-IDF scores...")
    tfidf_df = compute_tfidf(corpus, stop_words=stop_words)
    save_results(tfidf_output_path, tfidf_df)

    # Step 3: Load GSEA results
    print("[Step 3] Loading GSEA results...")
    gsea_results = pd.read_table(gsea_results_path)

    # Step 4: Run significance testing
    print("[Step 4] Running significance testing...")
    significant_terms_df = run_significance_tests(tfidf_df, gsea_results)
    save_results(significance_output_path, significant_terms_df)

    print("TF-IDF and significance analysis completed successfully.")
    #call: python3 -m scripts.analysis.gsea_refiner.tfidf_analysis