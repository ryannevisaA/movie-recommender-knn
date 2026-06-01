# 🎬 Movie Recommendation System — KNN Algorithm

A content-based movie recommendation system built with Python and scikit-learn using the **K-Nearest Neighbors (KNN)** algorithm.

---

## 📌 Overview

Given a movie you like, this system finds the most similar movies based on:
- **Genre similarity** (10 genre binary features)
- **IMDb rating**
- **Release year**
- **Popularity score**

The feature vectors are scaled with `MinMaxScaler` and KNN finds the nearest neighbors using **Euclidean distance** in the 13-dimensional feature space.

---

## 🗂️ Project Structure

```
movie_recommender/
├── movies.csv          # Dataset — 40 movies, 13 features
├── recommender.py      # Core KNN engine
├── visualize.py        # Analysis plots (PCA, heatmap, bar charts)
├── evaluate.py         # Evaluation metrics
├── app.py              # Interactive CLI
└── requirements.txt    # Dependencies
```

---

## ⚙️ How It Works

```
Movie Input
    │
    ▼
Feature Engineering
  ├── Genre binary flags  [Action, Comedy, Drama, ...]
  ├── Normalized Rating   [0–1]
  ├── Normalized Year     [0–1]
  └── Normalized Popularity [0–1]
    │
    ▼
MinMaxScaler (normalize all features)
    │
    ▼
KNN (NearestNeighbors, Euclidean distance)
    │
    ▼
Top-K Similar Movies
```

---

## 📊 Evaluation Results

| Metric              | Euclidean | Cosine | Manhattan |
|---------------------|-----------|--------|-----------|
| Genre Precision     | 99.0%     | 100%   | 98.0%     |
| Intra-list Diversity| 1.271     | 1.304  | 1.301     |
| Catalog Coverage    | 100%      | 97.5%  | 100%      |

---

## 🖼️ Sample Output

```
🎬 Top 5 Recommendations for 'Inception'
======================================================================
Rank  Title                   Year   Rating   Sim%    Genres
----------------------------------------------------------------------
1     The Matrix              1999   8.7      27.1    Action, Sci-Fi
2     Mad Max Fury Road       2015   8.1      23.4    Action, Sci-Fi
3     Blade Runner 2049       2017   8.0      17.6    Action, Drama, Sci-Fi
4     The Dark Knight         2008   9.0      2.3     Action, Thriller, Crime
5     Avengers Endgame        2019   8.4      0.0     Action, Sci-Fi, Fantasy
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/movie-recommender-knn.git
cd movie-recommender-knn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the interactive CLI
python app.py

# 4. Or run the core engine directly
python recommender.py

# 5. Generate analysis plots
python visualize.py

# 6. View evaluation metrics
python evaluate.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| scikit-learn | KNN model (`NearestNeighbors`) |
| pandas | Data loading & manipulation |
| numpy | Feature vector math |
| matplotlib | Visualization |
| seaborn | Correlation heatmap |

---

## 📈 Visualizations

- **Genre Distribution** — bar chart of movie genres in dataset
- **Rating Distribution** — histogram + rating vs popularity scatter
- **PCA Feature Space** — 2D projection of the 13D feature space
- **KNN Recommendations** — similarity bar chart for a query movie
- **Feature Correlation Heatmap** — seaborn heatmap
- **K Sensitivity Analysis** — how K affects precision vs diversity

---

## 🔮 Future Improvements

- [ ] Add collaborative filtering (user-item rating matrix)
- [ ] Use TF-IDF on movie plot descriptions
- [ ] Build a Streamlit web UI
- [ ] Integrate with TMDB API for real-time data
- [ ] Experiment with cosine similarity for sparse genre vectors

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---

## 📄 License

MIT License — free to use and modify.
