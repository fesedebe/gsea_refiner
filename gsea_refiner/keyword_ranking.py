import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ks_2samp

# Load corpus
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

# Run KS test
def run_significance_tests(tfidf_terms: pd.DataFrame, gsea_results: pd.DataFrame, pval_threshold: float = 0.05) -> pd.DataFrame:
    results = []

    for term in tfidf_terms["Term"]:
        test_pathways = gsea_results[gsea_results["pathway"].str.contains(term, case=False)]
        background_pathways = gsea_results[~gsea_results["pathway"].str.contains(term, case=False)]
        if len(test_pathways) < 2 or len(background_pathways) < 2:
            continue
        p_value = ks_2samp(test_pathways.index, background_pathways.index).pvalue
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