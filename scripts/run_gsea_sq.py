from gsea_refiner.categorization.gsea_sq import run_gsea_squared
import pandas as pd

def run_gsea_sq_pipeline(input_file: str, savename: str, catmap_file: str):
    df_catmap = pd.read_csv(catmap_file)
    categories = df_catmap["Category"].tolist()
    cat_terms = df_catmap["Regex"].tolist()

    return run_gsea_squared(
        df_gsea=input_file,
        categories=categories,
        cat_terms=cat_terms,
        savename=savename,
        verbose=True
    )

if __name__ == "__main__":
    input_file = "data/input/fGSEA_UCLAAllPatch_deseq_recur.txt"
    savename = "data/output/UCLAAllPatch_GSEAsq"
    catmap_file = "data/config/category_keywords.csv"

    run_gsea_sq_pipeline(input_file, savename, catmap_file)
