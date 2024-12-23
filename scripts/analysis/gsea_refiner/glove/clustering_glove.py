import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

def load_reduced_vectors(file_path):
    df = pd.read_csv(file_path)
    terms = df["Term"].tolist()
    vectors = df[["PC1", "PC2"]].values
    return terms, vectors

def kmeans_clustering(vectors, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    silhouette = silhouette_score(vectors, labels)
    ch_score = calinski_harabasz_score(vectors, labels)
    return labels, silhouette, ch_score

def dbscan_clustering(vectors, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(vectors)
    silhouette = silhouette_score(vectors, labels) if len(set(labels)) > 1 else -1
    ch_score = calinski_harabasz_score(vectors, labels) if len(set(labels)) > 1 else -1
    return labels, silhouette, ch_score

def plot_clusters(vectors, labels, terms, title, output_path=None):
    plt.figure(figsize=(12, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_points = vectors[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}", alpha=0.7)
    for i, term in enumerate(terms):
        plt.text(vectors[i, 0], vectors[i, 1], term, fontsize=8, alpha=0.7)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    input_file = "data/intermediate/reduced_term_vectors.csv"
    kmeans_output_file = "data/output/kmeans_clusters.csv"
    dbscan_output_file = "data/output/dbscan_clusters.csv"
    kmeans_plot = "results/figures/kmeans_plot.png"
    dbscan_plot = "results/figures/dbscan_plot.png"

    # Load vectors
    print("Loading reduced vectors...")
    terms, vectors = load_reduced_vectors(input_file)
    print(f"Loaded {len(terms)} terms.")

    # K-means clustering
    print("Performing k-means clustering...")
    n_clusters = 5
    kmeans_labels, kmeans_silhouette, kmeans_ch = kmeans_clustering(vectors, n_clusters)
    print(f"K-means Silhouette Score: {kmeans_silhouette:.2f}")
    print(f"K-means CH Score: {kmeans_ch:.2f}")

    kmeans_df = pd.DataFrame({"Term": terms, "Cluster": kmeans_labels})
    kmeans_df.to_csv(kmeans_output_file, index=False)
    print(f"K-means clusters saved to {kmeans_output_file}")

    plot_clusters(vectors, kmeans_labels, terms, "K-Means Clustering", kmeans_plot)

    # DBSCAN clustering
    print("Performing DBSCAN clustering...")
    eps = 0.5 
    min_samples = 5 
    dbscan_labels, dbscan_silhouette, dbscan_ch = dbscan_clustering(vectors, eps, min_samples)
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.2f}")
    print(f"DBSCAN CH Score: {dbscan_ch:.2f}")

    dbscan_df = pd.DataFrame({"Term": terms, "Cluster": dbscan_labels})
    dbscan_df.to_csv(dbscan_output_file, index=False)
    print(f"DBSCAN clusters saved to {dbscan_output_file}")

    plot_clusters(vectors, dbscan_labels, terms, "DBSCAN Clustering", dbscan_plot)