import os
from scripts.preprocessing.extract_names import process_gene_set_names
from scripts.preprocessing.tokenize_names import tokenize_corpus
from scripts.preprocessing.prepare_corpus import save_corpus_to_txt

def run_tokenize_pipeline(input_file: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    print("Extracting and cleaning pathway names...")
    cleaned_names = process_gene_set_names(input_file)
    print(f"Extracted and cleaned {len(cleaned_names)} pathway names.")
    
    print("Tokenizing pathway names...")
    tokenized_names = tokenize_corpus(cleaned_names)
    print(f"Tokenized {len(tokenized_names)} pathway names.")
    
    txt_output = os.path.join(output_dir, "corpus.txt")
    print("Saving tokenized pathway names...")
    save_corpus_to_txt(tokenized_names, txt_output)
    print(f"Saved corpus to {txt_output}.")
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_tokenize_pipeline(
        input_file="data/input/fGSEA_scn_HC2C5.txt", 
        output_dir="data/intermediate/"
    )