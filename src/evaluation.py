import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

def evaluate_clustering(X_scaled: np.ndarray, labels: np.ndarray, model_name: str = "Model") -> dict:
    """
    Calculates internal validation indices and smallest cluster proportion for a given model.
    """
    counts = pd.Series(labels).value_counts(normalize=True)
    return {
        "Model": model_name,
        "Silhouette": round(float(silhouette_score(X_scaled, labels)), 4),
        "Davies-Bouldin": round(float(davies_bouldin_score(X_scaled, labels)), 4),
        "Calinski-Harabasz": round(float(calinski_harabasz_score(X_scaled, labels)), 2),
        "Smallest_Cluster_%": round(float(counts.min() * 100), 2)
    }

def subsample_ari(model_factory, X: np.ndarray, n_runs: int = 10, frac: float = 0.8, seed: int = 42) -> str:
    """
    Calculates cluster perturbation stability via repeated subsampling (mean ± std).
    """
    rng = np.random.RandomState(seed)
    scores = []
    n_samples = len(X)
    sub_size = int(frac * n_samples)

    for _ in range(n_runs):
        idx1 = rng.choice(n_samples, sub_size, replace=False)
        idx2 = rng.choice(n_samples, sub_size, replace=False)
        common = np.intersect1d(idx1, idx2)

        if len(common) < 10:
            continue

        m1 = model_factory()
        m2 = model_factory()

        l1 = pd.Series(m1.fit_predict(X[idx1]), index=idx1).loc[common]
        l2 = pd.Series(m2.fit_predict(X[idx2]), index=idx2).loc[common]

        scores.append(adjusted_rand_score(l1, l2))

    if not scores:
        return "N/A"
    return f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"

def compute_rfm_baseline_labels(rfm_df: pd.DataFrame) -> np.ndarray:
    """
    Constructs a deterministic 4-cluster operational baseline using fixed thresholds.
    """
    r = pd.qcut(rfm_df["Recency"], q=4, labels=[4, 3, 2, 1]).astype(int)
    f = pd.cut(rfm_df["Frequency"], bins=[0, 1, 2, 4, np.inf], labels=[1, 2, 3, 4]).astype(int)
    m = pd.qcut(rfm_df["Monetary"], q=4, labels=[1, 2, 3, 4]).astype(int)
    total = (r + f + m).values
    baseline_labels = pd.cut(total, bins=[2, 5, 7, 9, 12], labels=[0, 1, 2, 3]).to_numpy().astype(int)
    return baseline_labels

def evaluate_models(X_scaled: np.ndarray, rfm_df: pd.DataFrame = None, k: int = 4, seeds: tuple = (42, 100, 2026)):
    """
    Evaluates baseline heuristic and candidate clustering algorithms.
    """
    results = []
    labels_dict = {}

    if rfm_df is not None:
        base_labels = compute_rfm_baseline_labels(rfm_df)
        labels_dict["RFM 4-Quantile Baseline"] = base_labels
        base_eval = evaluate_clustering(X_scaled, base_labels, "RFM 4-Quantile Baseline")
        base_eval["Stability_ARI (Subsample)"] = "Deterministic (Rule-based)"
        results.append(base_eval)

    km_factory = lambda: KMeans(n_clusters=k, random_state=seeds[0], n_init=10)
    km = km_factory()
    km_labels = km.fit_predict(X_scaled)
    labels_dict[f"K-Means (k={k})"] = km_labels
    km_eval = evaluate_clustering(X_scaled, km_labels, f"K-Means (k={k})")
    km_eval["Stability_ARI (Subsample)"] = subsample_ari(km_factory, X_scaled, seed=seeds[0])
    results.append(km_eval)

    agg_factory = lambda: AgglomerativeClustering(n_clusters=k)
    agg = agg_factory()
    agg_labels = agg.fit_predict(X_scaled)
    labels_dict[f"Agglomerative (k={k})"] = agg_labels
    agg_eval = evaluate_clustering(X_scaled, agg_labels, f"Agglomerative (k={k})")
    agg_eval["Stability_ARI (Subsample)"] = subsample_ari(agg_factory, X_scaled, seed=seeds[0])
    results.append(agg_eval)

    gmm_factory = lambda: GaussianMixture(n_components=k, random_state=seeds[0])
    gmm = gmm_factory()
    gmm_labels = gmm.fit_predict(X_scaled)
    labels_dict[f"Gaussian Mixture (k={k})"] = gmm_labels
    gmm_eval = evaluate_clustering(X_scaled, gmm_labels, f"Gaussian Mixture (k={k})")
    gmm_eval["Stability_ARI (Subsample)"] = subsample_ari(gmm_factory, X_scaled, seed=seeds[0])
    results.append(gmm_eval)

    results_df = pd.DataFrame(results)
    return results_df, labels_dict

def evaluate_k_range(X_scaled: np.ndarray, k_range: range = range(2, 9), seed: int = 42) -> pd.DataFrame:
    """
    Evaluates cluster validation indices and cluster proportions across k=2 to k=8.
    """
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_scaled)
        labels = km.labels_
        counts = pd.Series(labels).value_counts(normalize=True)
        records.append({
            "k": k,
            "Inertia": round(float(km.inertia_), 2),
            "Silhouette": round(float(silhouette_score(X_scaled, labels)), 4),
            "Davies-Bouldin": round(float(davies_bouldin_score(X_scaled, labels)), 4),
            "Calinski-Harabasz": round(float(calinski_harabasz_score(X_scaled, labels)), 2),
            "Smallest_Cluster_%": round(float(counts.min() * 100), 2)
        })
    return pd.DataFrame(records)
