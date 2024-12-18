import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_split_vectors(file_path):
    # Load terms and their 300d vectors from a file and return as arrays.
    terms = []
    vectors = []
    with open(file_path, "r") as f:
        for line in f:
            term, vector_str = line.split(":")
            terms.append(term.strip())
            vector = [float(v) for v in vector_str.strip(" []\n").split(",")]
            vectors.append(vector)
    return terms, np.array(vectors)

def reduce_dimensions(vectors, n_components=2):
    # Reduce high-dimensional vectors to a lower dimension (default: 2) using PCA.
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(vectors)
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    return reduced_vectors

def visualize_pca(terms, reduced_vectors, output_path=None, alpha=0.4, size=25, font_size=6):
    # Create a 2D scatter plot of terms using PCA-reduced vectors.
    plt.figure(figsize=(23, 17))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=alpha, edgecolor="k", s=size)
    for i, term in enumerate(terms):
        plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], term, fontsize=font_size, alpha=alpha)
    plt.title("PCA Dimensionality Reduction (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    if output_path:
        plt.savefig(output_path)
        print(f"PCA plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # File paths
    input_file = "data/intermediate/split_term_vectors.txt"
    output_file = "data/intermediate/reduced_term_vectors.csv"
    plot_file = "results/figures/pca_plot.png"

    # Load split term vectors
    print("Loading split term vectors...")
    terms, vectors = load_split_vectors(input_file)
    print(f"Loaded {len(terms)} terms.")

    # Reduce dimensions
    print("Reducing dimensions using PCA...")
    reduced_vectors = reduce_dimensions(vectors, n_components=2)

    # Save reduced vectors
    print(f"Saving reduced vectors to {output_file}...")
    reduced_df = pd.DataFrame(reduced_vectors, columns=["PC1", "PC2"])
    reduced_df["Term"] = terms
    reduced_df.to_csv(output_file, index=False)
    print(f"Reduced vectors saved to {output_file}.")

    # Visualize the reduced vectors
    print("Visualizing reduced vectors...")
    visualize_pca(terms, reduced_vectors, plot_file)
