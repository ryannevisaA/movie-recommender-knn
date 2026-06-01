"""
Movie Recommendation System using K-Nearest Neighbors (KNN)
============================================================
Author  : Your Name
Tools   : Python, scikit-learn, pandas, numpy
Dataset : Custom movie dataset (movies.csv)
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(filepath="movies.csv"):
    """Load movie dataset from CSV."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} movies with {len(df.columns)} features.")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

GENRE_COLS = ['Action','Comedy','Drama','Horror','Sci-Fi',
              'Romance','Thriller','Animation','Crime','Fantasy']

def build_feature_matrix(df):
    """
    Build the feature matrix for KNN.

    Features used:
      - Genre binary flags  (10 dimensions)
      - Normalized rating   (1 dimension)
      - Normalized year     (1 dimension)
      - Normalized popularity (1 dimension)

    Returns:
      X        : scaled feature matrix (numpy array)
      scaler   : fitted MinMaxScaler
      feature_names : list of feature names
    """
    feature_names = GENRE_COLS + ['rating', 'year', 'popularity']
    X_raw = df[feature_names].values.astype(float)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"[INFO] Feature matrix shape: {X_scaled.shape}")
    return X_scaled, scaler, feature_names


# ─────────────────────────────────────────────
# 3. KNN MODEL
# ─────────────────────────────────────────────

def train_knn(X, n_neighbors=6, metric='euclidean'):
    """
    Fit a KNN model.

    Parameters:
      n_neighbors : number of neighbors to find (K)
                    +1 because the movie itself is always the nearest
      metric      : distance metric ('euclidean', 'cosine', 'manhattan')
    """
    model = NearestNeighbors(
        n_neighbors=n_neighbors + 1,  # +1 to exclude self
        metric=metric,
        algorithm='brute'             # brute-force, best for small datasets
    )
    model.fit(X)
    print(f"[INFO] KNN model trained | K={n_neighbors} | metric={metric}")
    return model


# ─────────────────────────────────────────────
# 4. RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def recommend(movie_title, df, X, model, k=5):
    """
    Get top-K movie recommendations for a given title.

    Parameters:
      movie_title : string name of the query movie
      df          : original dataframe
      X           : scaled feature matrix
      model       : fitted NearestNeighbors model
      k           : number of recommendations

    Returns:
      DataFrame of recommended movies with similarity scores
    """
    titles = df['title'].str.lower().tolist()
    query = movie_title.strip().lower()

    # Find the movie index
    matches = [i for i, t in enumerate(titles) if query in t]
    if not matches:
        print(f"[ERROR] Movie '{movie_title}' not found in database.")
        print(f"        Available titles: {df['title'].tolist()}")
        return None

    idx = matches[0]
    matched_title = df.iloc[idx]['title']
    print(f"\n[QUERY] Finding movies similar to: '{matched_title}'")
    print("─" * 50)

    # Get K nearest neighbors
    distances, indices = model.kneighbors(X[idx].reshape(1, -1))

    # Exclude self (index 0 is always the movie itself)
    neighbor_indices = indices[0][1:k+1]
    neighbor_distances = distances[0][1:k+1]

    # Compute similarity score (0–100%)
    max_dist = neighbor_distances.max() if neighbor_distances.max() > 0 else 1
    similarity_scores = (1 - neighbor_distances / max_dist) * 100

    # Build result DataFrame
    results = df.iloc[neighbor_indices][['title', 'year', 'rating', 'popularity']].copy()
    results['distance'] = np.round(neighbor_distances, 4)
    results['similarity_%'] = np.round(similarity_scores, 1)
    results['genres'] = df.iloc[neighbor_indices].apply(
        lambda row: ", ".join([g for g in GENRE_COLS if row[g] == 1]), axis=1
    )
    results = results.reset_index(drop=True)
    results.index += 1  # 1-based ranking

    return matched_title, results


# ─────────────────────────────────────────────
# 5. DISPLAY RESULTS
# ─────────────────────────────────────────────

def display_recommendations(matched_title, results, k):
    """Pretty-print recommendation results."""
    print(f"\n🎬 Top {k} Recommendations for '{matched_title}'")
    print("=" * 70)
    print(f"{'Rank':<5} {'Title':<38} {'Year':<6} {'Rating':<8} {'Sim%':<7} {'Genres'}")
    print("-" * 70)
    for rank, row in results.iterrows():
        print(f"{rank:<5} {row['title']:<38} {int(row['year']):<6} {row['rating']:<8} {row['similarity_%']:<7} {row['genres']}")
    print("=" * 70)


# ─────────────────────────────────────────────
# 6. MAIN ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    df = load_data("movies.csv")

    # Build features
    X, scaler, feature_names = build_feature_matrix(df)

    # Train KNN
    K = 5
    model = train_knn(X, n_neighbors=K, metric='euclidean')

    # ── Demo: try different query movies ──
    demo_queries = ["Inception", "Toy Story", "The Notebook", "Get Out"]

    for query in demo_queries:
        result = recommend(query, df, X, model, k=K)
        if result:
            matched_title, recs = result
            display_recommendations(matched_title, recs, K)
            print()
