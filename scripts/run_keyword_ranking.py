import os
import pandas as pd
from gsea_refiner.keyword_ranking import load_corpus, compute_tfidf, run_significance_tests, save_results

def run_keyword_ranking_pipeline(corpus_path: str, gsea_results_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    stop_words = ["process", "pathway", "regulation", "response", "activity", "positive", "negative"]

    print("Computing TF-IDF scores...")
    corpus = load_corpus(corpus_path)
    tfidf_df = compute_tfidf(corpus, stop_words=stop_words)

    print("Running significance testing...")
    gsea_results = pd.read_table(gsea_results_path)
    significant_terms_df = run_significance_tests(tfidf_df, gsea_results)
    save_results(os.path.join(output_dir, "significant_terms.csv"), significant_terms_df)

    print("Completed")
    #call: python3 -m scripts.analysis.gsea_refiner.tfidf_analysis

if __name__ == "__main__":
    corpus_path = "data/intermediate/corpus.txt"
    gsea_results_path = "data/input/fGSEA_scn_HC2C5.txt"
    output_dir = "data/intermediate/"
    
    run_keyword_ranking_pipeline(corpus_path, gsea_results_path, output_dir)