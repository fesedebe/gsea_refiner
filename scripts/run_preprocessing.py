import os
from gsea_refiner.preprocessing.tokenize_corpus import (
    process_gene_set_names, tokenize_corpus, save_corpus_to_txt, save_corpus_to_json
)
from gsea_refiner.preprocessing.filter_and_weight import filter_and_weight_pathways

def process_and_save_corpus(input_file, output_txt, output_json=None):
    cleaned_names = process_gene_set_names(input_file)
    tokenized_names = tokenize_corpus(cleaned_names, stopwords=None)

    save_corpus_to_txt(tokenized_names, output_txt)
    if output_json:
        save_corpus_to_json(tokenized_names, output_json)

    print(f"Saved corpus to {output_txt}")

def filter_pathways(input_file, output_dir):
    filtered_output = os.path.join(output_dir, "filtered_pathways.csv")
    weighted_output = os.path.join(output_dir, "weighted_pathways.csv")

    filter_and_weight_pathways(
        input_file=input_file,
        output_filtered=filtered_output,
        output_weighted=weighted_output
    )

    print(f"Saved filtered pathways to {filtered_output}")
    print(f"Saved weighted scores to {weighted_output}")

    return filtered_output

def run_preprocessing_pipeline(input_file: str, output_dir: str, 
                               filter_pathways_flag: bool = True, save_json: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    # Process full corpus
    txt_output = os.path.join(output_dir, "corpus.txt")
    json_output = os.path.join(output_dir, "corpus.json") if save_json else None
    process_and_save_corpus(input_file, txt_output, json_output)

    # Process filtered corpus
    if filter_pathways_flag:
        filtered_input = filter_pathways(input_file, output_dir) 
        txt_filtered_output = os.path.join(output_dir, "corpus_filtered.txt")
        json_filtered_output = os.path.join(output_dir, "corpus_filtered.json") if save_json else None
        process_and_save_corpus(filtered_input, txt_filtered_output, json_filtered_output)

    print("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    input_file = "data/input/fGSEA_scn_HC2C5.txt"
    output_dir = "data/intermediate/"
    
    run_preprocessing_pipeline(input_file, output_dir, filter_pathways_flag=True, save_json=False)
