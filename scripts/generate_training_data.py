import pandas as pd
from gsea_refiner.utils import label_pathways_by_regex, clean_gene_set_name

def main():
    input_file = "data/input/gsea_scn.txt"
    catmap_file = "data/config/category_keywords.csv"
    output_file = "data/training/labeled_pathways.csv"

    df = pd.read_csv(input_file, sep="\t")
    df['pathway'] = df['pathway'].apply(clean_gene_set_name)
    df_catmap = pd.read_csv(catmap_file)
    categories = df_catmap["Category"].tolist()
    cat_terms = df_catmap["Regex"].tolist()

    labeled_df = label_pathways_by_regex(df, categories, cat_terms, col="pathway", label_col="label", keep_unlabeled=False)
    labeled_df[["pathway", "label"]].to_csv(output_file, index=False)
    print(f"Saved labeled training data to: {output_file}")

if __name__ == "__main__":
    main()
