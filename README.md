# 🎬 Movie Recommendation System — KNN Algorithm

A content-based movie recommendation system built with Python and scikit-learn using the **K-Nearest Neighbors (KNN)** algorithm.

---

## 📌 Over# 🎬 CineMatch — Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-ff4b4b?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?style=flat-square&logo=streamlit)](https://movie-recommender-knn-hhpt8wbfn7bzfintlpvduj.streamlit.app/)

A content-based movie recommendation system built with the **K-Nearest Neighbors (KNN)** algorithm. Select any movie and instantly get the most similar films based on genre, rating, release year, and popularity — visualized in a sleek dark-themed web UI.

🔗 **Live Demo:** [movie-recommender-knn.streamlit.app](https://movie-recommender-knn-hhpt8wbfn7bzfintlpvduj.streamlit.app/)

---

## ✨ Features

- 🎯 KNN-based content filtering across 40 movies and 10 genres
- 📐 3 distance metrics — Euclidean, Cosine, Manhattan
- 📊 PCA feature space visualization (13D → 2D)
- 🎛️ Adjustable K (3–10 recommendations)
- 📈 Similarity score bars and genre breakdown charts
- 🖥️ Interactive Streamlit web UI with dark cinema theme

---

## 🧠 How It Works

Each movie is represented as a **13-dimensional feature vector**:

| Feature | Description |
|---|---|
| Genre flags (×10) | Binary values for Action, Comedy, Drama, Horror, Sci-Fi, Romance, Thriller, Animation, Crime, Fantasy |
| Rating | Normalized IMDb rating (scaled 0–1) |
| Year | Normalized release year |
| Popularity | Normalized popularity score |

The KNN algorithm computes the distance between the selected movie's vector and every other movie in the dataset. The K closest movies (smallest distance) are returned as recommendations.

**Similarity score** is calculated as:
```
similarity = (1 - distance / max_distance) × 100
```

---

## 🗂️ Project Structure

```
movie-recommender-knn/
├── streamlit_app.py       # Main web UI (Streamlit)
├── recommender.py         # KNN engine — data loading, model training, recommendations
├── visualize.py           # Analysis plots — PCA, heatmap, genre distribution
├── evaluate.py            # Evaluation metrics — precision, coverage
├── app.py                 # Interactive CLI mode
├── movies.csv             # Dataset — 40 movies × 14 features
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 📊 Visualizations

| Plot | Description |
|---|---|
| PCA Feature Space | 13D movie vectors projected to 2D — selected movie highlighted |
| Similarity Scores | Horizontal bar chart of recommendation match % |
| Genre Distribution | Count of movies per genre in the dataset |
| Rating Distribution | IMDb rating histogram |
| K Sensitivity | How precision changes with different K values |
| Correlation Heatmap | Feature correlation matrix |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/ryannevisaA/movie-recommender-knn.git
cd movie-recommender-knn
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the web app**
```bash
python -m streamlit run streamlit_app.py
```

**4. Or run CLI mode**
```bash
python app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| scikit-learn | KNN algorithm, PCA, MinMaxScaler |
| Streamlit | Web UI framework |
| Pandas | Data loading and manipulation |
| NumPy | Feature vector computation |
| Matplotlib | Charts and visualizations |

---

## 📈 Evaluation Results

| Metric | Score |
|---|---|
| Genre Precision | ~99% |
| Catalog Coverage | 100% |
| Supported Metrics | Euclidean, Cosine, Manhattan |
| Dataset Size | 40 movies, 13 features |

---

## 🔮 Future Improvements

- [ ] Integrate TMDB API for real movie posters and descriptions
- [ ] Scale to MovieLens 100K dataset
- [ ] Add collaborative filtering (user-based recommendations)
- [ ] t-SNE visualization alongside PCA
- [ ] User rating input for personalized recommendations

---

## 👨‍💻 Author

**Ryan** — [GitHub](https://github.com/ryannevisaA)

---

## 📄 License

This project is licensed under the MIT License.
view

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

RYAN NEVISA A

- GitHub: @ryannevisaA
- LinkedIn: www.linkedin.com/in/ryan-nevisa-7171b6289

---

## 📄 License

MIT License — free to use and modify.
