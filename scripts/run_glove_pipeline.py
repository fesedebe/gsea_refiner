import os
from gsea_refiner.embeddings.glove_embeddings import map_corpus_to_vectors
from gsea_refiner.reduce_and_cluster import reduce_vectors, cluster_and_visualize

def run_glove_pipeline(glove_file, input_corpus, output_folder, kmeans=True, n_clusters = 40, dbscan=True, affinity_propagation=True):
    os.makedirs(output_folder, exist_ok=True)

    # 1: Map GloVe Embeddings
    print("Starting mapping of GloVe embeddings...")
    term_vectors_path = os.path.join(output_folder, "split_term_vectors.txt")
    map_corpus_to_vectors(glove_file, input_corpus, term_vectors_path)

    # 2: PCA Dimensionality Reduction
    print("Starting PCA...")
    reduced_vectors_path = os.path.join(output_folder, "reduced_term_vectors.csv")
    reduce_vectors(term_vectors_path, reduced_vectors_path)

    # 3: Clustering
    print("Starting clustering...")
    clusters_folder = os.path.join(output_folder, "clusters")
    os.makedirs(clusters_folder, exist_ok=True)

    cluster_and_visualize(
        reduced_vectors_path, clusters_folder, 
        kmeans=kmeans,  n_clusters = n_clusters,
        dbscan=dbscan, affinity_propagation=affinity_propagation
    )

    print("GloVe analysis completed. Outputs saved to:", output_folder)

if __name__ == "__main__":
    run_glove_pipeline(
        input_corpus = "data/intermediate/corpus.txt" , 
        glove_file = "data/input/glove.840B.300d.txt",
        output_folder = "data/intermediate/"
    )
