"""
Evaluation Module
=================
Evaluates the KNN recommender using:
  - Genre Precision (how many recommended movies share genres)
  - Intra-list Diversity
  - Coverage
  - Distance metric comparison
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from recommender import load_data, build_feature_matrix, train_knn, GENRE_COLS
import warnings
warnings.filterwarnings("ignore")


# ─── Genre Precision ──────────────────────────────────────────────────────────

def genre_precision(df, X, model, k=5):
    """
    For each movie, check what fraction of its K neighbors
    share at least one genre with it.
    """
    precisions = []
    for i in range(len(df)):
        dists, indices = model.kneighbors(X[i].reshape(1, -1))
        neighbor_idx = indices[0][1:k+1]

        query_genres = set(g for g in GENRE_COLS if df.iloc[i][g] == 1)
        hits = 0
        for ni in neighbor_idx:
            neighbor_genres = set(g for g in GENRE_COLS if df.iloc[ni][g] == 1)
            if query_genres & neighbor_genres:
                hits += 1
        precisions.append(hits / k)

    return np.mean(precisions)


# ─── Intra-list Diversity ─────────────────────────────────────────────────────

def intra_list_diversity(df, X, model, k=5):
    """
    Average pairwise distance among recommended items.
    Higher = more diverse recommendations.
    """
    diversities = []
    for i in range(len(df)):
        dists, indices = model.kneighbors(X[i].reshape(1, -1))
        neighbor_idx = indices[0][1:k+1]
        neighbor_vecs = X[neighbor_idx]

        pairwise_dists = []
        for a in range(len(neighbor_vecs)):
            for b in range(a+1, len(neighbor_vecs)):
                d = np.linalg.norm(neighbor_vecs[a] - neighbor_vecs[b])
                pairwise_dists.append(d)
        if pairwise_dists:
            diversities.append(np.mean(pairwise_dists))

    return np.mean(diversities)


# ─── Catalog Coverage ─────────────────────────────────────────────────────────

def catalog_coverage(df, X, model, k=5):
    """
    Fraction of all movies that appear in at least one recommendation list.
    """
    recommended = set()
    for i in range(len(df)):
        _, indices = model.kneighbors(X[i].reshape(1, -1))
        for ni in indices[0][1:k+1]:
            recommended.add(ni)
    return len(recommended) / len(df)


# ─── Metric Comparison ────────────────────────────────────────────────────────

def compare_metrics(df, X, k=5):
    """
    Compare KNN performance across different distance metrics.
    """
    metrics = ['euclidean', 'cosine', 'manhattan']
    results = []

    for metric in metrics:
        m = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='brute').fit(X)
        gp = genre_precision(df, X, m, k)
        ild = intra_list_diversity(df, X, m, k)
        cc = catalog_coverage(df, X, m, k)
        results.append({
            'Metric': metric.capitalize(),
            'Genre Precision': round(gp, 4),
            'Intra-list Diversity': round(ild, 4),
            'Catalog Coverage': round(cc, 4),
        })

    return pd.DataFrame(results)


# ─── K Sweep ──────────────────────────────────────────────────────────────────

def k_sweep(df, X, k_range=range(3, 11)):
    """
    Evaluate genre precision and diversity across different K values.
    """
    results = []
    for k in k_range:
        m = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='brute').fit(X)
        gp = genre_precision(df, X, m, k)
        ild = intra_list_diversity(df, X, m, k)
        results.append({'K': k, 'Genre Precision': round(gp, 4),
                         'Intra-list Diversity': round(ild, 4)})
    return pd.DataFrame(results)


# ─── Run Evaluation ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.chdir("/home/claude/movie_recommender")

    df = load_data("movies.csv")
    X, scaler, _ = build_feature_matrix(df)

    K = 5
    model = train_knn(X, n_neighbors=K, metric='euclidean')

    print("\n" + "="*55)
    print("  EVALUATION REPORT — KNN Movie Recommender")
    print("="*55)

    gp = genre_precision(df, X, model, K)
    ild = intra_list_diversity(df, X, model, K)
    cc = catalog_coverage(df, X, model, K)

    print(f"\n  K = {K} | Metric = Euclidean")
    print(f"  {'Genre Precision':<28} : {gp:.4f}  ({gp*100:.1f}%)")
    print(f"  {'Intra-list Diversity':<28} : {ild:.4f}")
    print(f"  {'Catalog Coverage':<28} : {cc:.4f}  ({cc*100:.1f}%)")

    print("\n── Distance Metric Comparison ─────────────────────")
    metric_df = compare_metrics(df, X, K)
    print(metric_df.to_string(index=False))

    print("\n── K Sensitivity ──────────────────────────────────")
    k_df = k_sweep(df, X)
    print(k_df.to_string(index=False))
    print("="*55)
