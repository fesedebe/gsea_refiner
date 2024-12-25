import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import plotly.express as px

def load_reduced_vectors(file_path):
    df = pd.read_csv(file_path)
    terms = df["Term"].tolist()
    vectors = df[["PC1", "PC2"]].values
    return terms, vectors

def perform_kmeans(vectors, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    silhouette = silhouette_score(vectors, labels)
    ch_score = calinski_harabasz_score(vectors, labels)
    return labels, silhouette, ch_score

def perform_dbscan(vectors, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(vectors)
    silhouette = silhouette_score(vectors, labels) if len(set(labels)) > 1 else -1
    ch_score = calinski_harabasz_score(vectors, labels) if len(set(labels)) > 1 else -1
    return labels, silhouette, ch_score

def perform_affinity_propagation(vectors, damping=0.9, preference=None):
    ap = AffinityPropagation(damping=damping, preference=preference, random_state=42)
    labels = ap.fit_predict(vectors)
    silhouette = silhouette_score(vectors, labels) if len(set(labels)) > 1 else -1
    ch_score = calinski_harabasz_score(vectors, labels) if len(set(labels)) > 1 else -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, silhouette, ch_score, n_clusters

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

def plot_interactive_clusters(vectors, labels, terms, title, output_path=None):
    df = pd.DataFrame({
        "PC1": vectors[:, 0],
        "PC2": vectors[:, 1],
        "Cluster": labels,
        "Term": terms
    })

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=["Term"],
        title=title,
        color_continuous_scale=px.colors.qualitative.Set1
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Interactive plot saved to {output_path}")
    else:
        fig.show()

def cluster_and_visualize(input_file, output_dir, kmeans=True,  n_clusters = 40, dbscan=True, affinity_propagation=True):
    print("Loading reduced vectors...")
    terms, vectors = load_reduced_vectors(input_file)
    print(f"Loaded {len(terms)} terms.")

    if kmeans:
        print("Performing K-means clustering...")
        kmeans_labels, kmeans_silhouette, kmeans_ch = perform_kmeans(vectors, n_clusters)
        print(f"K-means Silhouette Score: {kmeans_silhouette:.2f}")
        print(f"K-means CH Score: {kmeans_ch:.2f}")

        kmeans_output_file = f"{output_dir}/kmeans_clusters.csv"
        kmeans_df = pd.DataFrame({"Term": terms, "Cluster": kmeans_labels})
        kmeans_df.to_csv(kmeans_output_file, index=False)
        print(f"K-means clusters saved to {kmeans_output_file}")

        plot_interactive_clusters(vectors, kmeans_labels, terms, "K-Means Clustering", f"{output_dir}/kmeans_plot.html")

    if dbscan:
        print("Performing DBSCAN clustering...")
        eps = 0.5
        min_samples = 7
        dbscan_labels, dbscan_silhouette, dbscan_ch = perform_dbscan(vectors, eps, min_samples)
        print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.2f}")
        print(f"DBSCAN CH Score: {dbscan_ch:.2f}")

        dbscan_output_file = f"{output_dir}/dbscan_clusters.csv"
        dbscan_df = pd.DataFrame({"Term": terms, "Cluster": dbscan_labels})
        dbscan_df.to_csv(dbscan_output_file, index=False)
        print(f"DBSCAN clusters saved to {dbscan_output_file}")

        plot_interactive_clusters(vectors, dbscan_labels, terms, "DBSCAN Clustering", f"{output_dir}/dbscan_plot.html")

    if affinity_propagation:
        print("Performing Affinity Propagation clustering...")
        damping = 0.9
        preference = None
        ap_labels, ap_silhouette, ap_ch, n_clusters_ap = perform_affinity_propagation(vectors, damping, preference)
        print(f"Affinity Propagation Silhouette Score: {ap_silhouette:.2f}")
        print(f"Affinity Propagation CH Score: {ap_ch:.2f}")
        print(f"Number of clusters found: {n_clusters_ap}")

        affinity_output_file = f"{output_dir}/affinity_clusters.csv"
        ap_df = pd.DataFrame({"Term": terms, "Cluster": ap_labels})
        ap_df.to_csv(affinity_output_file, index=False)
        print(f"Affinity Propagation clusters saved to {affinity_output_file}")

        plot_interactive_clusters(vectors, ap_labels, terms, "Affinity Propagation Clustering", f"{output_dir}/affinity_plot.html")

if __name__ == "__main__":
    input_file = "data/intermediate/reduced_term_vectors.csv"
    output_dir = "data/output"

    cluster_and_visualize(input_file, output_dir, kmeans=True, dbscan=True, affinity_propagation=True)