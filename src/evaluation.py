from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd

def evaluate_clustering(X, labels, model_name: str) -> dict:
    """
    Evaluates unsupervised clustering models using silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
    """
    return {
        "Model": model_name,
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "Clusters": len(set(labels)) - (1 if -1 in labels else 0)
    }
