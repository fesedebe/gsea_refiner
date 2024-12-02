import os

def load_glove_embeddings(file_path):
    """
    Load pre-trained GloVe embeddings into a dictionary.

    Args:
        file_path (str): Path to the GloVe embedding file.
    
    Returns:
        dict: Mapping of words to their embedding vectors.
    """
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = list(map(float, values[1:]))
            embeddings[word] = vector
    return embeddings

def get_term_vectors(terms, embeddings):
    """
    Map terms to their corresponding embedding vectors.

    Args:
        terms (list): List of terms (keywords).
        embeddings (dict): Pre-loaded GloVe embeddings.
    
    Returns:
        dict: Mapping of terms to vectors.
    """
    term_vectors = {}
    for term in terms:
        if term in embeddings:
            term_vectors[term] = embeddings[term]
        else:
            print(f"Term '{term}' not found in embeddings.")
    return term_vectors

def handle_missing_terms(term, embeddings):
    """
    Handle terms not found in embeddings by splitting into words.
    
    Args:
        term (str): The missing term.
        embeddings (dict): Pre-loaded GloVe embeddings.
    
    Returns:
        list: Averaged vector for the term, or None if no words match.
    """
    words = term.split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if vectors:
        return [sum(dim) / len(vectors) for dim in zip(*vectors)]
    return None  # Skip if no words are found

if __name__ == "__main__":
    # File paths
    glove_file_path = "data/embeddings/glove.42B.300d.txt"
    significant_terms_file = "data/intermediate/significant_terms.csv"
    output_file = "data/intermediate/term_vectors.csv"

    # Load embeddings
    print("Loading GloVe embeddings...")
    embeddings = load_glove_embeddings(glove_file_path)
    print(f"Loaded {len(embeddings)} embeddings.")

    # Load significant terms
    print("Loading significant terms...")
    with open(significant_terms_file, "r") as f:
        terms = [line.strip() for line in f.readlines()]

    # Map terms to vectors
    print("Mapping terms to vectors...")
    term_vectors = {}
    for term in terms:
        vector = get_term_vectors([term], embeddings).get(term)
        if vector is None:
            vector = handle_missing_terms(term, embeddings)
        if vector is not None:
            term_vectors[term] = vector

    # Save term vectors
    print(f"Saving term vectors to {output_file}...")
    with open(output_file, "w") as f:
        for term, vector in term_vectors.items():
            f.write(f"{term}," + ",".join(map(str, vector)) + "\n")

    print("Embedding processing completed.")
