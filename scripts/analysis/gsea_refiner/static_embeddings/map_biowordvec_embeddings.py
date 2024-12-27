import fasttext
import os

def load_biowordvec_model(model_path):
    """
    Load the pre-trained BioWordVec model.
    Args:
        model_path (str): Path to the BioWordVec FastText model file.
    Returns:
        FastText model object.
    """
    print(f"Loading BioWordVec model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please download the BioWordVec model.")
    model = fasttext.load_model(model_path)
    print("BioWordVec model loaded successfully.")
    return model


def map_terms_to_embeddings(corpus_path, model, output_path):
    """
    Map terms in the corpus to their BioWordVec embeddings.
    Args:
        corpus_path (str): Path to the tokenized corpus file.
        model: Pre-trained FastText model object.
        output_path (str): Path to save the term embeddings.
    """
    print(f"Loading corpus from {corpus_path}...")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}. Please provide a valid corpus.")

    with open(corpus_path, "r") as f:
        terms = [line.strip() for line in f]

    term_embeddings = {}
    for term in terms:
        # Get the embedding for the term (or generate one for unseen terms)
        vector = model.get_word_vector(term)
        term_embeddings[term] = vector

    # Save term embeddings to a file
    print(f"Saving term embeddings to {output_path}...")
    with open(output_path, "w") as f:
        for term, vector in term_embeddings.items():
            vector_str = ",".join(map(str, vector))
            f.write(f"{term}: {vector_str}\n")
    print(f"Embeddings saved to {output_path}")


def map_biowordvec_pipeline(corpus_path, model_path, output_path):
    """
    Master function to map BioWordVec embeddings.
    Args:
        corpus_path (str): Path to the tokenized corpus file.
        model_path (str): Path to the BioWordVec FastText model file.
        output_path (str): Path to save the term embeddings.
    """
    # Load the pre-trained BioWordVec model
    model = load_biowordvec_model(model_path)

    # Map terms in the corpus to embeddings
    map_terms_to_embeddings(corpus_path, model, output_path)


if __name__ == "__main__":
    # Paths for testing
    corpus_path = "data/intermediate/tokenized_corpus.txt"
    model_path = "data/embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin"
    output_path = "data/output/biowordvec_term_embeddings.txt"

    # Test the BioWordVec pipeline
    print("Running BioWordVec pipeline...")
    map_biowordvec_pipeline(corpus_path, model_path, output_path)
    print("Pipeline completed.")
