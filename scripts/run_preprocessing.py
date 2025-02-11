import os
from gsea_refiner.preprocess import process_gene_set_names, tokenize_corpus, save_corpus_to_txt, save_corpus_to_json

def run_preprocessing_pipeline(input_file: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    cleaned_names = process_gene_set_names(input_file)
    print(f"Extracted and cleaned {len(cleaned_names)} pathway names.")

    tokenized_names = tokenize_corpus(cleaned_names, stopwords=None)
    print(f"Tokenized {len(tokenized_names)} pathway names.")

    txt_output = os.path.join(output_dir, "corpus.txt")
    json_output = os.path.join(output_dir, "corpus.json")

    save_corpus_to_txt(tokenized_names, txt_output)
    save_corpus_to_json(tokenized_names, json_output)
    print(f"Saved corpus to {txt_output} and {json_output}.")

    print("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    #input_file = "data/input/gsea_results.csv"
    input_file = "data/input/fGSEA_scn_HC2C5.txt"
    output_dir = "data/intermediate/"
    
    run_preprocessing_pipeline(input_file, output_dir)