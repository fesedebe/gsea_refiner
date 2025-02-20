def load_glove_embeddings(file_path):
    """
    Load pre-trained GloVe embeddings into a dictionary, where keys are words and values are corresponding vectors (300d floats).
    Format: {word: vector}.
    """
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            values = line.split()
            word = values[0]
            try:
                vector = list(map(float, values[1:]))
                embeddings[word] = vector
                # if len(embeddings) < 8:
                #     print(f"{word}: {vector[:7]} ...")
            except ValueError:
                print(f"Skipping corrupted line {line_num}: {line.strip()[:50]}")
                continue  
    return embeddings

def split_and_map_terms(terms, embeddings, phrase_level=False):
    # Map terms to embeddings as either individual word-level mappings (default) or multi-word phrase mappings.
    term_vectors = {}

    for term in terms:
        words = term.split()  

        if phrase_level:
            # Map entire term to a list of word vectors
            vectors = [embeddings[word] for word in words if word in embeddings]
            if vectors:
                term_vectors[term] = vectors
            else:
                print(f"No embeddings found for term '{term}'.")  
        else:
            # Map individual words to their vectors
            for word in words:
                if word in embeddings and word not in term_vectors:
                    term_vectors[word] = embeddings[word]
                elif word not in embeddings:
                    print(f"No embedding found for word '{word}'.") 

    return term_vectors

def map_corpus_to_vectors(glove_file, corpus_file_path, output_file, phrase_level=False):
    print("Loading GloVe embeddings...")
    embeddings = load_glove_embeddings(glove_file)
    print(f"Loaded {len(embeddings)} embeddings.")

    print("Loading tokenized corpus...")
    with open(corpus_file_path, "r") as f:
        terms = [line.strip() for line in f.readlines()]

    print("Processing corpus and mapping...")
    term_vectors = split_and_map_terms(terms, embeddings, phrase_level)

    print(f"Saving split word vectors to {output_file}...")
    with open(output_file, "w") as f:
        for term, vectors in term_vectors.items():
            f.write(f"{term}: {vectors}\n")

    print("Processing completed.")

if __name__ == "__main__":
    glove_file = "data/input/glove.840B.300d.txt"
    corpus_file_path = "data/intermediate/corpus.txt" 
    output_file = "data/intermediate/split_term_vectors.txt"

    map_corpus_to_vectors(glove_file, corpus_file_path, output_file)