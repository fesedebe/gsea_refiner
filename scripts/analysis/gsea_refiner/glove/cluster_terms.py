import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_reduced_vectors(file_path):
    """
    Load reduced term vectors and terms from a CSV file.
    """
    df = pd.read_csv(file_path)
    terms = df["Term"].tolist()
    vectors = df[["PC1", "PC2"]].values
    return terms, vectors

def cluster_vectors(vectors, n_clusters=5):
    """
    Perform k-means clustering on vectors and return cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(vectors)
    return cluster_labels, kmeans.cluster_centers_

def visualize_clusters(terms, vectors, cluster_labels, cluster_centers, output_path=None):
    """
    Plot clustered terms in 2D space with cluster centers.
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(vectors[:, 0], vectors[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7, s=50)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="red", marker="x", s=200, label="Cluster Centers")
    for i, term in enumerate(terms):
        plt.text(vectors[i, 0], vectors[i, 1], term, fontsize=8, alpha=0.6)
    plt.title("Clusters of Terms")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.legend()
    if output_path:
        plt.savefig(output_path)
        print(f"Cluster plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # File paths
    input_file = "data/intermediate/reduced_term_vectors.csv"
    output_file = "data/intermediate/clustered_terms.csv"
    plot_file = "results/figures/cluster_plot.png"

    # Load reduced vectors
    print("Loading reduced term vectors...")
    terms, vectors = load_reduced_vectors(input_file)
    print(f"Loaded {len(terms)} terms.")

    # Perform clustering
    print("Clustering vectors...")
    n_clusters = 5
    cluster_labels, cluster_centers = cluster_vectors(vectors, n_clusters=n_clusters)

    # Save clustered terms
    print(f"Saving clustered terms to {output_file}...")
    clustered_df = pd.DataFrame({"Term": terms, "Cluster": cluster_labels})
    clustered_df.to_csv(output_file, index=False)
    print(f"Clustered terms saved to {output_file}.")

    # Visualize clusters
    print("Visualizing clusters...")
    visualize_clusters(terms, vectors, cluster_labels, cluster_centers, plot_file)
