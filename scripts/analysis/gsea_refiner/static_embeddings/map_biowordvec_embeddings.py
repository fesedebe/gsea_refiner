import os
import fasttext
from gensim.models import KeyedVectors

def load_biowordvec_model(model_path, model_type="vector"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please download the BioWordVec model.")
    if model_type == "vector":
        model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    elif model_type == "model":
        model = fasttext.load_model(model_path)
    
    return model

def load_text_model(vec_file_path):
    print(f"Loading embeddings from text file: {vec_file_path}")
    embeddings = {}
    with open(vec_file_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0 and len(line.split()) == 2:
                continue
            parts = line.strip().split()
            word = parts[0]
            vector = list(map(float, parts[1:]))
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} embeddings from {vec_file_path}")
    return embeddings


def map_terms_to_embeddings(corpus_path, model, output_path):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}. Please provide a valid corpus.")

    with open(corpus_path, "r") as f:
        terms = [line.strip() for line in f]

    term_embeddings = {}
    for term in terms:
        # Get the embedding for the term (or generate one for unseen terms)
        vector = model.get_word_vector(term)
        term_embeddings[term] = vector

    with open(output_path, "w") as f:
        for term, vector in term_embeddings.items():
            vector_str = ",".join(map(str, vector))
            f.write(f"{term}: {vector_str}\n")

def map_biowordvec_pipeline(corpus_path, model_path, output_path):
    print(f"Loading BioWordVec model from {model_path}...")
    model = load_biowordvec_model(model_path)
    #model = load_text_model(model_path)
    print("BioWordVec model loaded successfully.")

    print(f"Loading corpus from {corpus_path}...")
    map_terms_to_embeddings(corpus_path, model, output_path)
    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    corpus_path = "data/intermediate/corpus.txt"
    model_path = "data/input/BioWordVec_PubMed_MIMICIII_d200.vec.bin"
    #model_path = "data/input/cc.en.300.bin" #temp testing for fasttext; works
    output_path = "data/intermediate/biowordvec_term_embeddings.txt"

    map_biowordvec_pipeline(corpus_path, model_path, output_path)
