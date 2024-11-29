import json
from typing import List

def save_corpus_to_txt(tokenized_names: List[List[str]], output_path: str) -> None:
    """
    Save tokenized pathway names to a text file, one pathway per line.
    
    Args:
        tokenized_names (List[List[str]]): List of tokenized pathway names.
        output_path (str): Path to save the text file.
    """
    with open(output_path, "w") as f:
        for tokens in tokenized_names:
            # Join tokens with a space and write to file
            f.write(" ".join(tokens) + "\n")

def save_corpus_to_json(tokenized_names: List[List[str]], output_path: str) -> None:
    """
    Save tokenized pathway names to a JSON file.
    
    Args:
        tokenized_names (List[List[str]]): List of tokenized pathway names.
        output_path (str): Path to save the JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(tokenized_names, f, indent=2)

if __name__ == "__main__":
    # Example 
    tokenized_names = [
        ["m", "phase"],
        ["g2m", "checkpoint"],
        ["atp", "hydrolysis", "activity"],
        ["chromosome", "segregation"],
        ["chromosomal", "region"]
    ]
    
    # Save to text and JSON formats
    save_corpus_to_txt(tokenized_names, "data/output/corpus.txt")
    save_corpus_to_json(tokenized_names, "data/output/corpus.json")
    
    print("Corpus saved to 'data/output/corpus.txt' and 'data/output/corpus.json'")
