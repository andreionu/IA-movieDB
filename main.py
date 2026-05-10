import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import Counter
import ast
import warnings
warnings.filterwarnings('ignore')

import requests as http_requests
import os
from flask import Flask, render_template, request, jsonify

SEP = "=" * 60

# ==============================================================
# 1. ALEGEREA SETULUI DE DATE DE PE KAGGLE
# ==============================================================
print(SEP)
print("1. ALEGEREA SETULUI DE DATE DE PE KAGGLE")
print(SEP)
print("Dataset: The Movies Dataset (Kaggle, Rounak Banik)")
print("Fisiere: movies_metadata.csv, credits.csv, keywords.csv,")
print("         ratings_small.csv, links_small.csv")
print()
print("Se incarca fisierele...")

df_movies   = pd.read_csv('movies_metadata.csv',  low_memory=False)
df_credits  = pd.read_csv('credits.csv')
df_keywords = pd.read_csv('keywords.csv')
df_ratings  = pd.read_csv('ratings_small.csv')
df_links    = pd.read_csv('links_small.csv')

print(f"  movies_metadata : {len(df_movies):,} randuri, {len(df_movies.columns)} coloane")
print(f"  credits         : {len(df_credits):,} randuri, {len(df_credits.columns)} coloane")
print(f"  keywords        : {len(df_keywords):,} randuri, {len(df_keywords.columns)} coloane")
print(f"  ratings_small   : {len(df_ratings):,} randuri, {len(df_ratings.columns)} coloane")
print(f"  links_small     : {len(df_links):,} randuri, {len(df_links.columns)} coloane")

# ==============================================================
# 2. PREGATIREA DATELOR
# ==============================================================
print(f"\n{SEP}")
print("2. PREGATIREA DATELOR")
print(SEP)

# --- Conversie ID-uri la int ---
for df, col in [(df_movies, 'id'), (df_credits, 'id'), (df_keywords, 'id')]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_movies   = df_movies.dropna(subset=['id']).copy()
df_credits  = df_credits.dropna(subset=['id']).copy()
df_keywords = df_keywords.dropna(subset=['id']).copy()

df_movies['id']   = df_movies['id'].astype(int)
df_credits['id']  = df_credits['id'].astype(int)
df_keywords['id'] = df_keywords['id'].astype(int)

df_links['tmdbId'] = pd.to_numeric(df_links['tmdbId'], errors='coerce')
df_links = df_links.dropna(subset=['tmdbId']).copy()
df_links['tmdbId'] = df_links['tmdbId'].astype(int)

# --- Curatare movies ---
df_movies = df_movies[df_movies['adult'] == 'False'].copy()
df_movies = df_movies[df_movies['title'].notna()].copy()
df_movies = df_movies.drop_duplicates(subset='id').copy()
df_movies['vote_average'] = pd.to_numeric(df_movies['vote_average'], errors='coerce')
df_movies['vote_count']   = pd.to_numeric(df_movies['vote_count'],   errors='coerce')
df_movies['poster_path']  = df_movies['poster_path'].fillna('')

# --- Functii helper pentru parsare JSON ---
def parse_json_field(val):
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except Exception:
        return []

def extract_names(val, key='name', limit=None):
    items = parse_json_field(val)
    names = [item[key] for item in items if isinstance(item, dict) and key in item]
    return names[:limit] if limit else names

def get_director(val):
    for p in parse_json_field(val):
        if isinstance(p, dict) and p.get('job') == 'Director':
            return p.get('name', '')
    return ''

# --- Parsare genuri, actori, regizori, cuvinte cheie ---
print("  Se parseaza genuri si metadate text...")
df_movies['genres_parsed'] = df_movies['genres'].apply(lambda x: extract_names(x))
df_credits['director']     = df_credits['crew'].apply(get_director)
df_credits['cast_top3']    = df_credits['cast'].apply(lambda x: extract_names(x, limit=3))
df_keywords['kw_list']     = df_keywords['keywords'].apply(lambda x: extract_names(x))

# --- Join: metadata + credits + keywords ---
df_full = (
    df_movies[['id', 'title', 'genres_parsed', 'vote_average',
               'vote_count', 'release_date', 'poster_path', 'overview']]
    .merge(df_credits[['id', 'director', 'cast_top3']], on='id', how='inner')
    .merge(df_keywords[['id', 'kw_list']],              on='id', how='inner')
)
df_full['year'] = pd.to_numeric(
    df_full['release_date'].str[:4], errors='coerce'
).fillna(0).astype(int)

df_full = df_full.drop_duplicates(subset='id').copy()
print(f"  Dupa join metadata+credits+keywords: {len(df_full):,} filme")
print(f"  Rating mediu TMDB : {df_full['vote_average'].mean():.2f}")
print(f"  Interval ani      : {df_full['year'][df_full['year']>0].min()} - {df_full['year'].max()}")
print(f"  Fara director     : {df_full['director'].eq('').sum():,} filme")

# --- Grafice exploratorii ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Explorarea Setului de Date — The Movies Dataset', fontsize=14, fontweight='bold')

axes[0].hist(df_full['vote_average'].dropna(), bins=25, color='steelblue', edgecolor='black')
axes[0].set_title('Distributia ratingurilor TMDB')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Numar filme')

all_genres = [g for sub in df_full['genres_parsed'] for g in sub]
gc = Counter(all_genres).most_common(10)
gn, gv = zip(*gc)
axes[1].barh(gn, gv, color='coral', edgecolor='black')
axes[1].set_title('Top 10 genuri')
axes[1].set_xlabel('Numar filme')
axes[1].invert_yaxis()

yr = df_full[df_full['year'] >= 1970]['year'].value_counts().sort_index()
axes[2].fill_between(yr.index, yr.values, color='mediumseagreen', alpha=0.7)
axes[2].set_title('Filme lansate pe an (din 1970)')
axes[2].set_xlabel('An')
axes[2].set_ylabel('Numar filme')

plt.tight_layout()
os.makedirs('static/charts', exist_ok=True)
plt.savefig('static/charts/data_exploration.png', dpi=150)
plt.close()
print("\n  Grafic salvat: 'static/charts/data_exploration.png'")

# ==============================================================
# 3. ALEGEREA ALGORITMULUI DE ML
# ==============================================================
print(f"\n{SEP}")
print("3. ALEGEREA ALGORITMULUI DE MACHINE LEARNING")
print(SEP)
print("Algoritm: SVD (Singular Value Decomposition) — Matrix Factorization")
print()
print("Justificare:")
print("  - ratings_small contine ~100k evaluari reale per utilizator")
print("  - SVD descompune matricea utilizator*film in vectori latenti care")
print("    captureaza preferinte implicite (genuri, stil, epoca etc.)")
print("  - Vectorii latenti ai filmelor permit similaritate item-item:")
print("    filme care atrag aceleasi tipuri de utilizatori sunt 'similare'")
print("  - Implementare: sklearn.decomposition.TruncatedSVD")
print("  - Validare: RMSE/MAE pe split 80/20 al rating-urilor")

# ==============================================================
# 4. ANTRENAREA MODELULUI
# ==============================================================
print(f"\n{SEP}")
print("4. ANTRENAREA MODELULUI")
print(SEP)

# --- Mapare movieId (MovieLens) → tmdbId → titlu ---
links_clean = df_links[['movieId', 'tmdbId']].copy()

df_rated = (df_ratings
    .merge(links_clean, on='movieId', how='inner')
    .merge(df_full[['id', 'title']], left_on='tmdbId', right_on='id', how='inner'))

print(f"  Rating-uri dupa mapare (movieId→tmdb→titlu): {len(df_rated):,}")
print(f"  Utilizatori unici : {df_rated['userId'].nunique():,}")
print(f"  Filme unice       : {df_rated['tmdbId'].nunique():,}")

# --- Matrice user-item ---
print("\n  Se construieste matricea user-item (userId x tmdbId)...")
pivot = df_rated.pivot_table(
    index='userId', columns='tmdbId', values='rating'
).fillna(0)
user_index = {uid: i for i, uid in enumerate(pivot.index)}
print(f"  Dimensiune matrice: {pivot.shape[0]} utilizatori x {pivot.shape[1]} filme")

# --- Normalizare: scadem media fiecarui user (doar pe itemele evaluate >0) ---
def normalize_user_means(matrix):
    """Centreaza ratingurile per user; returneaza (matrice_norm, medii_user)."""
    rated = (matrix > 0)
    counts = rated.sum(axis=1, keepdims=True)
    counts = np.where(counts == 0, 1, counts)
    means = matrix.sum(axis=1, keepdims=True) / counts
    normalized = np.where(rated, matrix - means, 0.0)
    return normalized, means.flatten()

pivot_vals   = pivot.values.copy()
pivot_norm, user_means_full = normalize_user_means(pivot_vals)

# --- TruncatedSVD pe intreaga matrice normalizata (pentru deployare) ---
N_COMPONENTS = 50
print(f"\n  Se aplica TruncatedSVD (n_components={N_COMPONENTS}) pe matrice normalizata...")
svd_full = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
U_full   = svd_full.fit_transform(pivot_norm)   # (n_users, 50)
Vt_full  = svd_full.components_                  # (50, n_films)

var_explained = svd_full.explained_variance_ratio_.sum() * 100
print(f"  Varianta explicata de {N_COMPONENTS} componente: {var_explained:.1f}%")

# --- Precomputare top-50 filme similare per film (item-item in spatiu latent) ---
movie_ids_in_matrix = pivot.columns.tolist()  # lista tmdbId
Vt_T = Vt_full.T                               # (n_films, 50) — vectori latenti filme

print("  Se calculeaza similaritatile item-item in spatiu latent...")
sim_matrix = cosine_similarity(Vt_T)           # (n_films, n_films)

TOP_SIM = 50
precomputed_recs = {}
for i, tmdb_id in enumerate(movie_ids_in_matrix):
    sims = sim_matrix[i]
    top_idxs = sims.argsort()[::-1]
    recs_list = []
    for j in top_idxs:
        if i == j:
            continue
        recs_list.append((movie_ids_in_matrix[j], float(sims[j])))
        if len(recs_list) == TOP_SIM:
            break
    precomputed_recs[tmdb_id] = recs_list

del sim_matrix  # eliberam memoria
print(f"  SVD: top-{TOP_SIM} recomandari precomputate pentru {len(precomputed_recs):,} filme")

# --- TF-IDF Content-Based (al doilea algoritm, pentru comparatie) ---
print("\n  Se construieste modelul TF-IDF Content-Based (pentru comparatie)...")

df_cb = df_full[df_full['id'].isin(set(movie_ids_in_matrix))].copy().reset_index(drop=True)

def make_soup(row):
    genres   = ' '.join(row['genres_parsed'] or [])
    keywords = ' '.join(row['kw_list'] or [])
    director = (row['director'] or '').replace(' ', '')
    cast     = ' '.join([n.replace(' ', '') for n in (row['cast_top3'] or [])])
    return f"{genres} {keywords} {director} {director} {cast}"

df_cb['soup'] = df_cb.apply(make_soup, axis=1)

tfidf_vec = TfidfVectorizer(stop_words='english', max_features=10_000, ngram_range=(1, 2))
tfidf_mat = tfidf_vec.fit_transform(df_cb['soup'].fillna(''))
cb_id_to_idx = pd.Series(df_cb.index, index=df_cb['id']).drop_duplicates().to_dict()

print("  Se calculeaza similaritatile TF-IDF item-item...")
tfidf_sim = cosine_similarity(tfidf_mat)

precomputed_recs_cb = {}
for i in range(len(df_cb)):
    sims     = tfidf_sim[i]
    top_idxs = sims.argsort()[::-1]
    rlist    = []
    for j in top_idxs:
        if i == j:
            continue
        rlist.append((int(df_cb.iloc[j]['id']), float(sims[j])))
        if len(rlist) == TOP_SIM:
            break
    precomputed_recs_cb[int(df_cb.iloc[i]['id'])] = rlist

del tfidf_sim
print(f"  TF-IDF: top-{TOP_SIM} recomandari precomputate pentru {len(precomputed_recs_cb):,} filme")

# --- Index de cautare: titlu → tmdbId, tmdbId → metadate ---
title_to_tmdb       = df_full.set_index('title')['id'].to_dict()
title_to_tmdb_lower = {k.lower(): v for k, v in title_to_tmdb.items()}
tmdb_to_info        = df_full.set_index('id').to_dict(orient='index')

# ==============================================================
# 5. VALIDAREA MODELULUI
# ==============================================================
print(f"\n{SEP}")
print("5. VALIDAREA MODELULUI")
print(SEP)
print("Split 80% antrenare / 20% testare pe rating-urile mapate")
print()

train_df, test_df = train_test_split(df_rated, test_size=0.2, random_state=42)
print(f"  Set antrenare : {len(train_df):,} ratinguri")
print(f"  Set testare   : {len(test_df):,} ratinguri")

pivot_train = train_df.pivot_table(
    index='userId', columns='tmdbId', values='rating'
).fillna(0)

pivot_train_norm, user_means_train = normalize_user_means(pivot_train.values)

svd_val  = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
U_train  = svd_val.fit_transform(pivot_train_norm)
Vt_train = svd_val.components_
pred_norm_val = np.dot(U_train, Vt_train)
# Denormalizare: adaugam inapoi media fiecarui user
pred_denorm = pred_norm_val + user_means_train[:, np.newaxis]
pred_df = pd.DataFrame(pred_denorm,
                        index=pivot_train.index,
                        columns=pivot_train.columns)

y_true, y_pred = [], []
for _, row in test_df.iterrows():
    uid, tmid, true_r = row['userId'], row['tmdbId'], row['rating']
    if uid in pred_df.index and tmid in pred_df.columns:
        y_true.append(true_r)
        y_pred.append(pred_df.loc[uid, tmid])

y_true, y_pred = np.array(y_true), np.array(y_pred)
rmse_model    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
mae_model     = float(np.mean(np.abs(y_true - y_pred)))
global_mean   = float(train_df['rating'].mean())
rmse_baseline = float(np.sqrt(mean_squared_error(y_true, [global_mean] * len(y_true))))
improvement   = (rmse_baseline - rmse_model) / rmse_baseline * 100

print(f"  RMSE model SVD   : {rmse_model:.4f}")
print(f"  MAE  model SVD   : {mae_model:.4f}")
print(f"  RMSE baseline    : {rmse_baseline:.4f}  (prezice mereu media = {global_mean:.2f})")
print(f"  Imbunatatire     : {improvement:.1f}% fata de baseline")
print(f"  Evaluari prezise : {len(y_true):,} din {len(test_df):,} ({len(y_true)/len(test_df)*100:.1f}%)")

# --- Jaccard Similarity pentru ambii algoritmi ---
print("\n  Se calculeaza Jaccard Similarity pentru SVD si TF-IDF...")
np.random.seed(42)
JACCARD_SAMPLE = 150
sample_ids = [int(df_cb.iloc[i]['id'])
              for i in np.random.choice(len(df_cb), JACCARD_SAMPLE, replace=False)]

jaccard_svd_scores, jaccard_cb_scores = [], []
for tid in sample_ids:
    q_genres = set(tmdb_to_info.get(tid, {}).get('genres_parsed', []) or [])
    for recs_src, scores_list in [(precomputed_recs, jaccard_svd_scores),
                                   (precomputed_recs_cb, jaccard_cb_scores)]:
        if tid not in recs_src:
            continue
        for rec_id, _ in recs_src[tid][:5]:
            r_genres = set(tmdb_to_info.get(rec_id, {}).get('genres_parsed', []) or [])
            union = q_genres | r_genres
            inter = q_genres & r_genres
            scores_list.append(len(inter) / len(union) if union else 0.0)

jaccard_svd = float(np.mean(jaccard_svd_scores)) if jaccard_svd_scores else 0.0
jaccard_cb  = float(np.mean(jaccard_cb_scores))  if jaccard_cb_scores else 0.0
print(f"  Jaccard SVD    : {jaccard_svd:.4f}")
print(f"  Jaccard TF-IDF : {jaccard_cb:.4f}")

# --- Grafic validare complet ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Rezultatele Validarii Modelului', fontsize=13, fontweight='bold')

axes[0].bar(['Baseline\n(medie)', f'SVD\n(n={N_COMPONENTS})'],
            [rmse_baseline, rmse_model],
            color=['lightcoral', 'steelblue'], edgecolor='black', width=0.4)
axes[0].set_title('RMSE: SVD vs. Baseline')
axes[0].set_ylabel('RMSE (mai mic = mai bun)')
axes[0].set_ylim(0, rmse_baseline * 1.3)
for i, v in enumerate([rmse_baseline, rmse_model]):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

axes[1].scatter(y_true[:500], y_pred[:500], alpha=0.3, color='steelblue', s=10)
axes[1].plot([0.5, 5.5], [0.5, 5.5], 'r--', linewidth=1.5, label='Predictie perfecta')
axes[1].set_xlabel('Rating real')
axes[1].set_ylabel('Rating prezis de SVD')
axes[1].set_title('Rating real vs. prezis (esantion 500)')
axes[1].legend()

axes[2].bar(['SVD\n(Collaborative)', 'TF-IDF\n(Content-Based)'],
            [jaccard_svd, jaccard_cb],
            color=['steelblue', 'mediumseagreen'], edgecolor='black', width=0.4)
axes[2].set_title('Jaccard Similarity pe genuri\n(coerenta recomandarilor)')
axes[2].set_ylabel('Jaccard (mai mare = mai bun)')
axes[2].set_ylim(0, max(jaccard_svd, jaccard_cb) * 1.4)
for i, v in enumerate([jaccard_svd, jaccard_cb]):
    axes[2].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('static/charts/validation_results.png', dpi=150)
plt.close()
print("\n  Grafic salvat: 'static/charts/validation_results.png'")

# ==============================================================
# 6. UTILIZAREA MODELULUI PENTRU PREDICTII
# ==============================================================
print(f"\n{SEP}")
print("6. UTILIZAREA MODELULUI PENTRU PREDICTII")
print(SEP)


def get_similar_movies(title: str, n: int = 10):
    """Top-N filme similare bazate pe spatiul latent SVD (item-item)."""
    tmdb_id = title_to_tmdb_lower.get(title.lower())
    if tmdb_id is None:
        close = [t for t in title_to_tmdb if title.lower() in t.lower()][:5]
        sugestii = ', '.join(f'"{t}"' for t in close) if close else 'niciun titlu similar'
        return None, f"Filmul nu a fost gasit. Sugestii: {sugestii}"
    if tmdb_id not in precomputed_recs:
        return None, (f"'{title}' exista in metadata dar nu are "
                      f"rating-uri in dataset.")
    top = precomputed_recs[tmdb_id][:n]
    results = []
    for rec_id, score in top:
        info = tmdb_to_info.get(rec_id, {})
        if not info:
            continue
        poster = info.get('poster_path', '')
        results.append({
            'tmdb_id':      rec_id,
            'title':        info.get('title', 'N/A'),
            'year':         int(info.get('year', 0)),
            'genres':       [g for g in info.get('genres_parsed', []) if g],
            'vote_average': round(float(info.get('vote_average') or 0), 1),
            'poster_path':  poster if poster and str(poster) != 'nan' else '',
            'similarity':   round(score, 4),
        })
    return results, None


def get_user_recommendations(user_id: int, n: int = 10):
    """Top-N filme recomandate pentru un utilizator (SVD predictii)."""
    if user_id not in pivot.index:
        return None, f"Utilizatorul {user_id} nu exista in dataset."
    user_idx = user_index[user_id]
    predicted = np.dot(U_full[user_idx], Vt_full) + user_means_full[user_idx]
    seen_ids  = set(df_rated[df_rated['userId'] == user_id]['tmdbId'])

    results = []
    for j in predicted.argsort()[::-1]:
        tid = movie_ids_in_matrix[j]
        if tid in seen_ids:
            continue
        info = tmdb_to_info.get(tid, {})
        if not info:
            continue
        poster = info.get('poster_path', '')
        results.append({
            'tmdb_id':          tid,
            'title':            info.get('title', 'N/A'),
            'year':             int(info.get('year', 0)),
            'genres':           [g for g in info.get('genres_parsed', []) if g],
            'vote_average':     round(float(info.get('vote_average') or 0), 1),
            'poster_path':      poster if poster and str(poster) != 'nan' else '',
            'predicted_rating': round(float(predicted[j]), 2),
        })
        if len(results) == n:
            break
    return results, None


def get_cb_recommendations(title: str, n: int = 10):
    """Top-N filme similare bazate pe TF-IDF Content-Based (genuri+cuvinte cheie+regizor)."""
    tmdb_id = title_to_tmdb_lower.get(title.lower())
    if tmdb_id is None:
        close = [t for t in title_to_tmdb if title.lower() in t.lower()][:5]
        sugestii = ', '.join(f'"{t}"' for t in close) if close else 'niciun titlu similar'
        return None, f"Filmul nu a fost gasit. Sugestii: {sugestii}"
    if tmdb_id not in precomputed_recs_cb:
        return None, f"'{title}' nu are date de continut disponibile."
    top = precomputed_recs_cb[tmdb_id][:n]
    results = []
    for rec_id, score in top:
        info = tmdb_to_info.get(rec_id, {})
        if not info:
            continue
        poster = info.get('poster_path', '')
        results.append({
            'tmdb_id':      rec_id,
            'title':        info.get('title', 'N/A'),
            'year':         int(info.get('year', 0)),
            'genres':       [g for g in info.get('genres_parsed', []) if g],
            'vote_average': round(float(info.get('vote_average') or 0), 1),
            'poster_path':  poster if poster and str(poster) != 'nan' else '',
            'similarity':   round(score, 4),
        })
    return results, None


# Teste rapide
print("  Test get_similar_movies('Inception', n=5):")
recs_test, err = get_similar_movies('Inception', n=5)
if err:
    print(f"    {err}")
else:
    for r in recs_test:
        print(f"    {r['title']} ({r['year']}) — sim: {r['similarity']:.3f}")

print()
print("  Test get_user_recommendations(1, n=5):")
urecs_test, uerr = get_user_recommendations(1, n=5)
if uerr:
    print(f"    {uerr}")
else:
    for r in urecs_test:
        print(f"    {r['title']} ({r['year']}) — rating prezis: {r['predicted_rating']:.2f}")

# ==============================================================
# 7. PREZENTAREA REZULTATELOR — Aplicatie Flask
# ==============================================================
print(f"\n{SEP}")
print("7. PREZENTAREA REZULTATELOR — Aplicatie Flask")
print(SEP)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

TMDB_API_KEY  = "0a6dca1e18d2d0c01590ab01163b14c3"
TMDB_IMG      = "https://image.tmdb.org/t/p/w342"
POSTER_CACHE  = os.path.join(os.path.dirname(__file__), 'static', 'posters')
os.makedirs(POSTER_CACHE, exist_ok=True)

_api_path_cache: dict = {}   # tmdb_id -> poster_path proaspat din API


def _fetch_poster(tmdb_id: int, fallback_path: str = ''):
    """Returneaza URL local al posterei (descarcata daca lipseste).
    Foloseste TMDB API v3 pentru a obtine poster_path curent, apoi descarca imaginea."""
    local_file = os.path.join(POSTER_CACHE, f"{tmdb_id}.jpg")
    if os.path.exists(local_file):
        return f"/static/posters/{tmdb_id}.jpg"

    # Obtine poster_path proaspat din TMDB API (o singura data per film)
    if tmdb_id not in _api_path_cache:
        try:
            resp = http_requests.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                params={'api_key': TMDB_API_KEY},
                timeout=6
            )
            if resp.status_code == 200:
                _api_path_cache[tmdb_id] = resp.json().get('poster_path') or ''
            else:
                _api_path_cache[tmdb_id] = fallback_path
        except Exception:
            _api_path_cache[tmdb_id] = fallback_path

    poster_path = _api_path_cache[tmdb_id]
    if not poster_path:
        return None

    # Descarca si salveaza imaginea local
    try:
        img = http_requests.get(TMDB_IMG + poster_path, timeout=8)
        if img.status_code == 200 and img.headers.get('Content-Type', '').startswith('image'):
            with open(local_file, 'wb') as f:
                f.write(img.content)
            return f"/static/posters/{tmdb_id}.jpg"
    except Exception:
        pass
    return None


def _enrich_poster(recs):
    """Adauga poster_url (local) la fiecare recomandare."""
    for r in recs:
        r['poster_url'] = _fetch_poster(r.get('tmdb_id'), r.get('poster_path', ''))
    return recs


STATS = {
    'rmse':          round(rmse_model,    4),
    'mae':           round(mae_model,     4),
    'rmse_baseline': round(rmse_baseline, 4),
    'improvement':   round(improvement,   1),
    'jaccard_svd':   round(jaccard_svd,   4),
    'jaccard_cb':    round(jaccard_cb,    4),
    'n_films':       len(precomputed_recs),
    'n_users':       int(df_rated['userId'].nunique()),
    'n_ratings':     len(df_rated),
    'var_explained': round(var_explained, 1),
    'n_components':  N_COMPONENTS,
}


@app.route('/')
def index():
    return render_template('index.html',
                           recommendations=None,
                           query='', error=None,
                           stats=STATS)


@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form.get('title', '').strip()
    recs, err = get_similar_movies(title, n=10)
    if err:
        return render_template('index.html',
                               recommendations=None,
                               error=err, query=title,
                               stats=STATS)
    return render_template('index.html',
                           recommendations=_enrich_poster(recs),
                           query=title, error=None,
                           stats=STATS)


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    title, svd_recs, cb_recs, error = '', None, None, None
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        svd_recs, err1 = get_similar_movies(title, n=8)
        cb_recs,  err2 = get_cb_recommendations(title, n=8)
        error = err1 or err2
        if svd_recs: _enrich_poster(svd_recs)
        if cb_recs:  _enrich_poster(cb_recs)
    return render_template('compare.html',
                           title=title,
                           svd_recs=svd_recs,
                           cb_recs=cb_recs,
                           error=error,
                           stats=STATS)


@app.route('/methodology')
def methodology():
    return render_template('methodology.html', stats=STATS)


@app.route('/api/recommend')
def api_recommend():
    title = request.args.get('title', '').strip()
    n = min(int(request.args.get('n', 10)), 50)
    recs, err = get_similar_movies(title, n=n)
    if err:
        return jsonify({'error': err}), 404
    return jsonify(_enrich_poster(recs))


@app.route('/api/titles')
def api_titles():
    q = request.args.get('q', '').strip().lower()
    if len(q) < 2:
        return jsonify([])
    matches = sorted(t for t in title_to_tmdb if q in t.lower())[:20]
    return jsonify(matches)


print(f"  RMSE model    : {rmse_model:.4f}")
print(f"  Filme indexate: {len(precomputed_recs):,}")
print()
print("  Aplicatia porneste la http://127.0.0.1:5000")
print("  Apasa CTRL+C pentru a opri serverul.")
print(SEP)

app.run(debug=False, host='0.0.0.0', port=5000)
