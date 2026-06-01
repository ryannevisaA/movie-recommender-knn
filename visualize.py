"""
Visualization Module
====================
Generates analysis plots saved as PNG files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from recommender import load_data, build_feature_matrix, train_knn, recommend, GENRE_COLS
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f9f9f9',
})

PALETTE = ['#185FA5','#1D9E75','#D85A30','#BA7517','#993556',
           '#534AB7','#3B6D11','#993C1D','#0F6E56','#854F0B']


# ─── Plot 1: Genre Distribution ───────────────────────────────────────────────

def plot_genre_distribution(df, save_path="plot_genre_distribution.png"):
    counts = df[GENRE_COLS].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index, counts.values,
                   color=PALETTE[:len(counts)], edgecolor='white', height=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                str(int(val)), va='center', fontsize=10)
    ax.set_xlabel("Number of Movies", fontsize=11)
    ax.set_title("Genre Distribution in Dataset", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlim(0, counts.max() + 3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")


# ─── Plot 2: Rating Distribution ──────────────────────────────────────────────

def plot_rating_distribution(df, save_path="plot_rating_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Histogram
    axes[0].hist(df['rating'], bins=12, color='#185FA5', edgecolor='white', alpha=0.85)
    axes[0].set_xlabel("IMDb Rating", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Rating Distribution", fontsize=12, fontweight='bold')

    # Rating vs Popularity scatter
    scatter = axes[1].scatter(df['rating'], df['popularity'],
                               c=df['year'], cmap='viridis', s=60, alpha=0.8, edgecolors='white')
    plt.colorbar(scatter, ax=axes[1], label='Year')
    axes[1].set_xlabel("IMDb Rating", fontsize=11)
    axes[1].set_ylabel("Popularity Score", fontsize=11)
    axes[1].set_title("Rating vs Popularity (colored by Year)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")


# ─── Plot 3: PCA 2D Feature Space ─────────────────────────────────────────────

def plot_pca_feature_space(df, X, save_path="plot_pca_feature_space.png"):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Color by dominant genre
    def dominant_genre(row):
        for g in GENRE_COLS:
            if row[g] == 1:
                return g
        return 'Other'

    df['dominant_genre'] = df.apply(dominant_genre, axis=1)
    genres_present = df['dominant_genre'].unique()
    color_map = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(genres_present)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for genre in genres_present:
        mask = df['dominant_genre'] == genre
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color_map[genre], label=genre, s=70, alpha=0.85, edgecolors='white')

    # Annotate a few key movies
    highlight = ['Inception', 'Toy Story', 'The Notebook', 'The Dark Knight', 'Spirited Away']
    for i, row in df.iterrows():
        if row['title'] in highlight:
            ax.annotate(row['title'], (X_pca[i, 0], X_pca[i, 1]),
                        fontsize=8, xytext=(5, 5), textcoords='offset points', color='#333')

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("PCA: Movie Feature Space (2D Projection)", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")


# ─── Plot 4: KNN Recommendation for a Query ───────────────────────────────────

def plot_knn_recommendations(query_title, df, X, model, k=5,
                              save_path="plot_knn_recommendations.png"):
    result = recommend(query_title, df, X, model, k=k)
    if result is None:
        return
    matched_title, recs = result

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(recs))]
    bars = ax.barh(recs['title'][::-1], recs['similarity_%'][::-1],
                   color=colors[::-1], edgecolor='white', height=0.6)
    for bar, val in zip(bars, recs['similarity_%'][::-1]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va='center', fontsize=10)

    ax.set_xlabel("Similarity Score (%)", fontsize=11)
    ax.set_title(f"Top {k} KNN Recommendations for '{matched_title}'",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlim(0, 115)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")


# ─── Plot 5: Feature Correlation Heatmap ──────────────────────────────────────

def plot_correlation_heatmap(df, save_path="plot_correlation_heatmap.png"):
    cols = GENRE_COLS + ['rating', 'year', 'popularity']
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")


# ─── Plot 6: K vs Diversity ───────────────────────────────────────────────────

def plot_k_sensitivity(df, X, save_path="plot_k_sensitivity.png"):
    """Show how average distance changes with K (elbow analysis)."""
    from sklearn.neighbors import NearestNeighbors

    k_values = list(range(2, 16))
    avg_distances = []

    for k in k_values:
        m = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
        dists, _ = m.kneighbors(X)
        avg_distances.append(dists[:, 1:].mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_values, avg_distances, marker='o', color='#185FA5',
            linewidth=2, markersize=6)
    ax.fill_between(k_values, avg_distances, alpha=0.1, color='#185FA5')
    ax.set_xlabel("K (Number of Neighbors)", fontsize=11)
    ax.set_ylabel("Average Euclidean Distance", fontsize=11)
    ax.set_title("K Sensitivity Analysis — Choosing Optimal K", fontsize=13,
                 fontweight='bold', pad=12)
    ax.set_xticks(k_values)

    # Annotate optimal K
    best_k = k_values[np.argmin(np.gradient(avg_distances))]
    ax.axvline(best_k, color='#D85A30', linestyle='--', alpha=0.7, label=f'Suggested K={best_k}')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {save_path}")


# ─── RUN ALL ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    df = load_data("movies.csv")
    X, scaler, feature_names = build_feature_matrix(df)
    model = train_knn(X, n_neighbors=5)

    print("\n[INFO] Generating all visualization plots...\n")
    plot_genre_distribution(df)
    plot_rating_distribution(df)
    plot_pca_feature_space(df, X)
    plot_knn_recommendations("Inception", df, X, model)
    plot_correlation_heatmap(df)
    plot_k_sensitivity(df, X)
    print("\n[DONE] All plots saved.")
