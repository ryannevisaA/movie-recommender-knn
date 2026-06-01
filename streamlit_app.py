import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="CineMatch — KNN Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #0c0c0f;
    }

    .block-container {
        padding: 2rem 2.5rem 2rem 2.5rem;
        max-width: 1200px;
    }

    section[data-testid="stSidebar"] {
        background-color: #111116;
        border-right: 1px solid #1e1e28;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.5rem;
    }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        color: #ffffff;
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin: 0;
    }

    .hero-accent {
        color: #ff6b6b;
    }

    .hero-sub {
        font-size: 0.9rem;
        color: #555566;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 500;
        margin-top: 0.5rem;
    }

    .stat-card {
        background: #111116;
        border: 1px solid #1e1e28;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }

    .stat-num {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #ff6b6b;
        line-height: 1;
    }

    .stat-lbl {
        font-size: 0.75rem;
        color: #555566;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }

    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #555566;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .selected-movie-card {
        background: #111116;
        border: 1px solid #1e1e28;
        border-left: 3px solid #ff6b6b;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
    }

    .movie-title-big {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: #ffffff;
        margin: 0 0 0.4rem 0;
    }

    .movie-meta {
        font-size: 0.85rem;
        color: #555566;
        margin-bottom: 0.8rem;
    }

    .genre-chip {
        display: inline-block;
        background: #1a1a24;
        border: 1px solid #2a2a38;
        color: #aaaacc;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 5px;
        margin-bottom: 5px;
        text-transform: uppercase;
        font-weight: 500;
    }

    .rec-card {
        background: #111116;
        border: 1px solid #1e1e28;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 10px;
        transition: border-color 0.2s;
    }

    .rec-card:hover {
        border-color: #2a2a3e;
    }

    .rec-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        background: #1a1a24;
        border: 1px solid #2a2a38;
        border-radius: 50%;
        font-size: 0.75rem;
        color: #ff6b6b;
        font-weight: 600;
        margin-right: 10px;
        vertical-align: middle;
    }

    .rec-title {
        font-size: 0.95rem;
        color: #e0e0ee;
        font-weight: 500;
        vertical-align: middle;
    }

    .rec-year {
        font-size: 0.8rem;
        color: #555566;
        margin-left: 8px;
        vertical-align: middle;
    }

    .sim-bar-bg {
        background: #1a1a24;
        border-radius: 3px;
        height: 4px;
        margin: 8px 0 6px 0;
    }

    .sim-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #ff6b6b, #ff9f9f);
    }

    .sim-pct {
        font-size: 0.78rem;
        color: #ff6b6b;
        font-weight: 600;
    }

    .dist-val {
        font-size: 0.78rem;
        color: #444455;
        margin-left: 10px;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        color: #777788 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }

    div[data-testid="stSelectbox"] > div > div {
        background-color: #1a1a24 !important;
        border-color: #2a2a38 !important;
        color: #ccccdd !important;
        border-radius: 8px !important;
    }

    .stSlider > div > div > div {
        background-color: #ff6b6b !important;
    }

    hr {
        border-color: #1e1e28 !important;
    }

    .sidebar-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .sidebar-sub {
        font-size: 0.72rem;
        color: #444455;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1.5rem;
    }

    .how-it-works {
        background: #0e0e14;
        border: 1px solid #1a1a24;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
    }

    .how-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #444455;
        font-weight: 600;
        margin-bottom: 0.7rem;
    }

    .how-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        color: #666677;
        line-height: 1.5;
    }

    .how-dot {
        width: 5px;
        height: 5px;
        background: #ff6b6b;
        border-radius: 50%;
        margin-right: 8px;
        margin-top: 6px;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)

GENRE_COLS = ['Action','Comedy','Drama','Horror','Sci-Fi',
              'Romance','Thriller','Animation','Crime','Fantasy']

@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

@st.cache_resource
def build_model(metric, k):
    df = load_data()
    feature_names = GENRE_COLS + ['rating', 'year', 'popularity']
    X = df[feature_names].values.astype(float)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='brute')
    model.fit(X_scaled)
    return X_scaled, model

def get_recommendations(movie_idx, X, model, df, k):
    distances, indices = model.kneighbors(X[movie_idx].reshape(1, -1))
    neighbor_idx  = indices[0][1:k+1]
    neighbor_dist = distances[0][1:k+1]
    max_dist = neighbor_dist.max() if neighbor_dist.max() > 0 else 1
    sim_scores = (1 - neighbor_dist / max_dist) * 100
    results = df.iloc[neighbor_idx].copy()
    results['distance']   = np.round(neighbor_dist, 4)
    results['similarity'] = np.round(sim_scores, 1)
    results['genres_list'] = results.apply(lambda r: [g for g in GENRE_COLS if r[g] == 1], axis=1)
    return results.reset_index(drop=True)

df = load_data()


with st.sidebar:
    st.markdown('<div class="sidebar-title">CineMatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">KNN Recommendation Engine</div>', unsafe_allow_html=True)

    selected_movie = st.selectbox("Select movie", df['title'].tolist(), label_visibility="visible")
    k = st.slider("Recommendations (K)", min_value=3, max_value=10, value=5)
    metric = st.selectbox("Distance metric", ['euclidean', 'cosine', 'manhattan'])

    metric_desc = {
        'euclidean': 'Straight-line distance in 13D feature space.',
        'cosine':    'Angle between vectors — direction over magnitude.',
        'manhattan': 'Sum of absolute differences per feature.'
    }

    st.markdown(f"""
    <div class="how-it-works">
        <div class="how-title">How it works</div>
        <div class="how-item"><div class="how-dot"></div>Each movie = 13-number vector (genres + rating + year + popularity)</div>
        <div class="how-item"><div class="how-dot"></div>KNN finds K closest movies in that feature space</div>
        <div class="how-item"><div class="how-dot"></div><em style="color:#555566">{metric_desc[metric]}</em></div>
    </div>
    """, unsafe_allow_html=True)


X, model = build_model(metric, k)
movie_idx = df[df['title'] == selected_movie].index[0]
recs = get_recommendations(movie_idx, X, model, df, k)
query_movie = df.iloc[movie_idx]
query_genres = [g for g in GENRE_COLS if query_movie[g] == 1]


st.markdown("""
<div style="margin-bottom: 2rem;">
    <div class="hero-title">Cine<span class="hero-accent">Match</span></div>
    <div class="hero-sub">K-Nearest Neighbors · Content-Based Filtering · scikit-learn</div>
</div>
""", unsafe_allow_html=True)


c1, c2, c3, c4 = st.columns(4)
for col, num, lbl in zip([c1,c2,c3,c4], ['40','10','13','KNN'], ['Movies','Genres','Features','Algorithm']):
    with col:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


chips = "".join([f'<span class="genre-chip">{g}</span>' for g in query_genres])
st.markdown(f"""
<div class="selected-movie-card">
    <div class="section-label">Selected movie</div>
    <div class="movie-title-big">{query_movie['title']}</div>
    <div class="movie-meta">{int(query_movie['year'])} &nbsp;·&nbsp; ★ {query_movie['rating']} IMDb &nbsp;·&nbsp; Popularity {int(query_movie['popularity'])}/100</div>
    <div>{chips}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f'<div class="section-label">Top {k} recommendations</div>', unsafe_allow_html=True)


left_col, right_col = st.columns([3, 2])

with left_col:
    for i, row in recs.iterrows():
        chips_rec = "".join([f'<span class="genre-chip">{g}</span>' for g in row['genres_list']])
        sim = int(row['similarity'])
        st.markdown(f"""
        <div class="rec-card">
            <div style="margin-bottom:6px">
                <span class="rec-rank">{i+1}</span>
                <span class="rec-title">{row['title']}</span>
                <span class="rec-year">{int(row['year'])} · ★ {row['rating']}</span>
            </div>
            <div>{chips_rec}</div>
            <div class="sim-bar-bg"><div class="sim-bar-fill" style="width:{sim}%"></div></div>
            <span class="sim-pct">{sim}% match</span>
            <span class="dist-val">dist: {row['distance']}</span>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    
    fig2, ax2 = plt.subplots(figsize=(4, 4.2))
    fig2.patch.set_facecolor('#111116')
    ax2.set_facecolor('#111116')

    titles_short = [t[:20]+'…' if len(t)>20 else t for t in recs['title'][::-1]]
    sims = recs['similarity'][::-1]

    bars = ax2.barh(titles_short, sims, color='#ff6b6b', height=0.45)
    for bar, val in zip(bars, sims):
        ax2.text(val+1, bar.get_y()+bar.get_height()/2,
                 f'{val:.0f}%', va='center', fontsize=8, color='#888899')

    ax2.set_xlim(0, 120)
    ax2.set_xlabel("Similarity %", color='#444455', fontsize=9)
    ax2.tick_params(colors='#555566', labelsize=8)
    ax2.xaxis.label.set_color('#444455')
    for spine in ax2.spines.values():
        spine.set_color('#1e1e28')
        spine.set_linewidth(0.5)

    plt.tight_layout(pad=1.2)
    st.pyplot(fig2, use_container_width=True)
    plt.close()


st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Feature space visualization (PCA — 13D → 2D)</div>', unsafe_allow_html=True)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

fig3, ax3 = plt.subplots(figsize=(11, 5))
fig3.patch.set_facecolor('#0c0c0f')
ax3.set_facecolor('#0c0c0f')

rec_indices = recs.index.tolist()

for i in range(len(df)):
    if i == movie_idx:
        continue
    if i in rec_indices:
        ax3.scatter(X_pca[i,0], X_pca[i,1], c='#ff6b6b', s=80, alpha=0.9, zorder=3)
        ax3.annotate(df.iloc[i]['title'], (X_pca[i,0], X_pca[i,1]),
                     fontsize=7, color='#ff6b6b', xytext=(5,5), textcoords='offset points')
    else:
        ax3.scatter(X_pca[i,0], X_pca[i,1], c='#222230', s=40, alpha=1, zorder=2,
                    edgecolors='#2a2a3a', linewidths=0.5)

ax3.scatter(X_pca[movie_idx,0], X_pca[movie_idx,1],
            c='#ffffff', s=140, zorder=5, edgecolors='#ff6b6b', linewidths=2)
ax3.annotate(f'  {selected_movie}', (X_pca[movie_idx,0], X_pca[movie_idx,1]),
             fontsize=9, color='#ffffff', fontweight='bold',
             xytext=(6,6), textcoords='offset points')

var = pca.explained_variance_ratio_
ax3.set_xlabel(f"PC1  ({var[0]*100:.1f}% variance)", color='#333344', fontsize=9)
ax3.set_ylabel(f"PC2  ({var[1]*100:.1f}% variance)", color='#333344', fontsize=9)
ax3.tick_params(colors='#333344', labelsize=8)
for spine in ax3.spines.values():
    spine.set_color('#1a1a24')
    spine.set_linewidth(0.5)

legend_elements = [
    mpatches.Patch(facecolor='#ffffff', edgecolor='#ff6b6b', label='Selected'),
    mpatches.Patch(facecolor='#ff6b6b', label='Recommended'),
    mpatches.Patch(facecolor='#222230', edgecolor='#2a2a3a', label='Other'),
]
ax3.legend(handles=legend_elements, facecolor='#111116', edgecolor='#1e1e28',
           labelcolor='#666677', fontsize=8, loc='upper right')

plt.tight_layout(pad=1.5)
st.pyplot(fig3, use_container_width=True)
plt.close()

st.markdown("---")
st.markdown('<div style="font-size:0.72rem;color:#333344;text-align:center;letter-spacing:0.08em">CINEMATCH &nbsp;·&nbsp; PYTHON &nbsp;·&nbsp; SCIKIT-LEARN &nbsp;·&nbsp; STREAMLIT &nbsp;·&nbsp; KNN</div>', unsafe_allow_html=True)
