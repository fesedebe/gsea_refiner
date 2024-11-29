import os
from scripts.preprocessing.extract_names import process_gene_set_names
from scripts.preprocessing.tokenize_names import tokenize_corpus
from scripts.preprocessing.prepare_corpus import save_corpus_to_txt, save_corpus_to_json

def run_tokenize_pipeline(input_file: str, output_dir: str) -> None:
    """
    Master script to run the tokenization preprocessing pipeline:
    
    Args:
        input_file (str): Path to the GSEA results file.
        output_dir (str): Directory to save the output files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract and clean pathway names
    print("[Step 1] Extracting and cleaning pathway names...")
    cleaned_names = process_gene_set_names(input_file)
    print(f"Extracted and cleaned {len(cleaned_names)} pathway names.")
    
    # Step 2: Tokenize pathway names
    print("[Step 2] Tokenizing pathway names...")
    tokenized_names = tokenize_corpus(cleaned_names)
    print(f"Tokenized {len(tokenized_names)} pathway names.")
    
    # Step 3: Save corpus to files
    txt_output = os.path.join(output_dir, "corpus.txt")
    json_output = os.path.join(output_dir, "corpus.json")
    print("[Step 3] Saving tokenized pathway names...")
    save_corpus_to_txt(tokenized_names, txt_output)
    save_corpus_to_json(tokenized_names, json_output)
    print(f"Saved corpus to {txt_output} and {json_output}.")
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_tokenize_pipeline(
        input_file="data/input/fGSEA_UCLAAllPatch_deseq_recur.txt", 
        output_dir="data/intermediate/"
    )
#call: python3 -m scripts.preprocessing.run_tokenize_pipeline