"""
Interactive CLI — Movie Recommendation System
=============================================
Run this file to interactively get movie recommendations.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from recommender import load_data, build_feature_matrix, train_knn, recommend, display_recommendations

BANNER = """
╔══════════════════════════════════════════════════════╗
║       🎬 Movie Recommendation System (KNN)           ║
║       Built with Python + scikit-learn               ║
╚══════════════════════════════════════════════════════╝
"""

def main():
    print(BANNER)

    # Setup
    df = load_data("movies.csv")
    X, scaler, feature_names = build_feature_matrix(df)

    # Ask user for K and metric
    print("\nConfiguration:")
    try:
        k = int(input("  Enter K (number of recommendations) [default=5]: ").strip() or 5)
        metric = input("  Distance metric — euclidean / cosine / manhattan [default=euclidean]: ").strip() or "euclidean"
    except (ValueError, EOFError):
        k, metric = 5, "euclidean"

    model = train_knn(X, n_neighbors=k, metric=metric)

    print("\n  Available movies:")
    for i, title in enumerate(df['title'].tolist(), 1):
        print(f"    {i:>2}. {title}")

    # Main interaction loop
    while True:
        print("\n" + "─"*50)
        query = input("  Enter a movie name (or 'quit' to exit): ").strip()
        if query.lower() in ('quit', 'q', 'exit'):
            print("\n  Goodbye! 🎬\n")
            break

        result = recommend(query, df, X, model, k=k)
        if result:
            matched_title, recs = result
            display_recommendations(matched_title, recs, k)

        another = input("\n  Try another movie? (y/n) [default=y]: ").strip().lower()
        if another == 'n':
            print("\n  Goodbye! 🎬\n")
            break


if __name__ == "__main__":
    main()
