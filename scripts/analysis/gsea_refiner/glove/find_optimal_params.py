import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
#from scripts.utils.clustering_utils import load_reduced_vectors
from run_clustering_glove import load_reduced_vectors

def find_optimal_k(vectors, max_k=20):
    silhouette_scores = []
    k_values = range(2, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(vectors)
        silhouette = silhouette_score(vectors, labels)
        silhouette_scores.append((k, silhouette))

    # Find the best k based on the highest silhouette score
    best_k, best_silhouette = max(silhouette_scores, key=lambda x: x[1])
    return {"k": best_k, "silhouette_score": best_silhouette}

def find_best_dbscan_params(vectors, eps_values, min_samples_values):
    best_params = {"eps": None, "min_samples": None, "silhouette_score": -1}
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(vectors)
            if len(set(labels)) > 1:  # At least 2 clusters required for silhouette
                silhouette = silhouette_score(vectors, labels)
                if silhouette > best_params["silhouette_score"]:
                    best_params.update({
                        "eps": eps,
                        "min_samples": min_samples,
                        "silhouette_score": silhouette
                    })
    return best_params

if __name__ == "__main__":
    # Load vectors
    input_file = "data/intermediate/reduced_term_vectors.csv"
    print("Loading reduced vectors...")
    terms, vectors = load_reduced_vectors(input_file)

    # Find optimal k for k-means
    print("Finding optimal k for k-means...")
    k_result = find_optimal_k(vectors, max_k=20)
    print(f"Best k: {k_result['k']} with Silhouette Score: {k_result['silhouette_score']:.2f}")

    # Find best DBSCAN parameters
    print("Finding best parameters for DBSCAN...")
    eps_values = [0.3, 0.5, 0.7]
    min_samples_values = [3, 5, 10]
    dbscan_result = find_best_dbscan_params(vectors, eps_values, min_samples_values)
    print(f"Best DBSCAN params: eps={dbscan_result['eps']}, min_samples={dbscan_result['min_samples']} with Silhouette Score: {dbscan_result['silhouette_score']:.2f}")
