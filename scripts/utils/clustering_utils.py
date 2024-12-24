import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_reduced_vectors(file_path):
    df = pd.read_csv(file_path)
    terms = df["Term"].tolist()
    vectors = df[["PC1", "PC2"]].values
    return terms, vectors

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