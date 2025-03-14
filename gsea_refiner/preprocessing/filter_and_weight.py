import pandas as pd
from gsea_refiner.utils import read_file
from gsea_refiner.preprocessing.tokenize_corpus import process_and_save_corpus

def filter_and_weight_pathways(input_file, output_filtered, output_weighted, nes_col="NES", pathway_col="pathway", 
    pval_col="padj", nes_threshold=2.2, pval_threshold=0.0000005)-> None:
    """Filters pathways by statistical significance (p-adj) and biological importance (NES). Both can be adjusted or disabled."""

    df = read_file(input_file)

    for col in [nes_col, pathway_col, pval_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input file.")
        
    #Apply filtering
    filter_conditions = []
    if nes_threshold is not None:
        filter_conditions.append(df[nes_col].abs() >= nes_threshold)
    if pval_threshold is not None:
        filter_conditions.append(df[pval_col] <= pval_threshold)

    df_filtered = df if not filter_conditions else df.loc[pd.concat(filter_conditions, axis=1).all(axis=1)]
    tokenized_df = process_and_save_corpus(input_file, save=False)
    df_filtered = df_filtered.merge(tokenized_df, on="pathway", how="left")
    df_filtered.to_csv(output_filtered, index=False)

    #Compute weights
    pathway_counts_all = df[pathway_col].value_counts()
    pathway_counts_sig = df_filtered[pathway_col].value_counts()

    weight_scores = (pathway_counts_sig / pathway_counts_all).fillna(0).reset_index()
    weight_scores.columns = [pathway_col, "Weight_Score"]
    weight_scores.to_csv(output_weighted, index=False)

    print(f"Saved {len(df_filtered)} filtered pathways to: {output_filtered}")
    print(f"Saved weighted pathway scores to: {output_weighted}")