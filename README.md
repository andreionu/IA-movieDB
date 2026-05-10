## Cum se ruleaza proiectul

### Windows (PowerShell)

```powershell
.venv\Scripts\activate
python main.py
# Aplicatia devine disponibila la http://127.0.0.1:5000
```

### Linux / Mac

**Prima oara (setup):**
```bash
# Instaleaza dependente sistem (Ubuntu/Debian)
sudo apt install python3 python3-pip python3-venv

# Creeaza virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib scikit-learn flask requests
```

**La fiecare rulare:**
```bash
source .venv/bin/activate
python main.py
# Aplicatia devine disponibila la http://127.0.0.1:5000
```

> Daca `python` nu e recunoscut, foloseste `python3` in loc.

Prima rulare dureaza ~2-3 minute (antrenare SVD + TF-IDF + precomputare similaritati).
Restartat rapid deoarece posterele sunt cache-uite local in `static/posters/`.

## Instalare dependente

```bash
pip install pandas numpy matplotlib scikit-learn flask requests
```

## Fisiere CSV necesare (The Movies Dataset — Kaggle)

Toate fisierele trebuie sa fie in **radacina proiectului** (langa `main.py`):

| Fisier | Continut |
|--------|---------|
| `movies_metadata.csv` | Metadate film: titlu, genuri, overview, poster_path, vote_average |
| `credits.csv` | Actori si echipa (JSON) per film |
| `keywords.csv` | Cuvinte cheie (JSON) per film |
| `ratings_small.csv` | ~100k rating-uri (userId, movieId, rating) |
| `links_small.csv` | Mapare movieId (MovieLens) ↔ tmdbId |

Dataset: The Movies Dataset (Rounak Banik, Kaggle).

## Arhitectura

### `main.py` — pipeline secvential (fisier unic)

Ruleaza cei 7 pasi in ordine, apoi porneste Flask. Nu exista module separate.

| Pas | Ce face |
|-----|---------|
| 1 | Incarca cele 5 CSV-uri, printeaza dimensiunile |
| 2 | Parseaza JSON (genuri, actori, regizor, cuvinte cheie), face join → `df_full` (45k filme), construieste matricea user-item normalizata |
| 3 | Explica alegerea algoritmilor (SVD + TF-IDF) |
| 4 | Antreneaza `TruncatedSVD(n_components=50)` pe matricea 671×9025; construieste si modelul TF-IDF content-based; precompute top-50 similaritati item-item pentru ambii algoritmi |
| 5 | Valideaza pe split 80/20: RMSE, MAE, Jaccard; salveaza grafice in `static/charts/` |
| 6 | Testeaza `get_similar_movies()`, `get_user_recommendations()`, `get_cb_recommendations()` |
| 7 | Porneste Flask pe portul 5000 |

### Rute Flask

| Ruta | Metoda | Descriere |
|------|--------|-----------|
| `/` | GET | Homepage cu search SVD si bara de statistici |
| `/recommend` | POST | Returneaza top-10 filme similare via SVD |
| `/compare` | GET, POST | Comparatie SVD vs. TF-IDF side-by-side |
| `/methodology` | GET | Documentatie cei 7 pasi + grafice |
| `/api/recommend` | GET | API JSON: `?title=Inception&n=10` |
| `/api/titles` | GET | Autocomplete: `?q=inc` → lista titluri (max 20) |

### Fluxul de ID-uri
```
ratings_small.movieId  →  links_small.tmdbId  →  movies_metadata.id (tmdbId)
      (MovieLens ID)         (mapare)                (TMDB ID)
```
Toate structurile interne folosesc **tmdbId** ca cheie.

### Structuri cheie din memorie (dupa antrenare)

| Variabila | Tip | Continut |
|-----------|-----|---------|
| `pivot` | DataFrame | `userId × tmdbId` cu rating-uri (fillna=0) |
| `user_index` | dict | `{userId: index_int}` — lookup O(1) pentru user recommendations |
| `U_full`, `Vt_full` | ndarray | Factori SVD antrenati pe toate datele |
| `user_means_full` | ndarray | Media per user pentru denormalizare predictii |
| `precomputed_recs` | dict | `{tmdbId: [(tmdbId_rec, score), ...]}` top-50 SVD per film |
| `precomputed_recs_cb` | dict | `{tmdbId: [(tmdbId_rec, score), ...]}` top-50 TF-IDF per film |
| `title_to_tmdb` | dict | `{titlu: tmdbId}` cautare exacta |
| `title_to_tmdb_lower` | dict | `{titlu_lower: tmdbId}` cautare case-insensitive |
| `tmdb_to_info` | dict | `{tmdbId: {title, year, genres_parsed, poster_path, ...}}` |

### Structura fisiere

```
IA-Movies/
├── main.py                  # Pipeline + Flask (fisier unic)
├── static/
│   ├── style.css            # CSS comun pentru toate paginile
│   ├── charts/
│   │   ├── data_exploration.png    # Generat la rulare
│   │   └── validation_results.png  # Generat la rulare
│   └── posters/             # Cache postere TMDB (descarcare automata)
│       └── {tmdb_id}.jpg
└── templates/
    ├── index.html       # Homepage — search SVD, carduri filme
    ├── compare.html     # Comparatie SVD vs. TF-IDF side-by-side
    └── methodology.html # Documentatie metodologie + grafice
```

### Template-uri

Toate folosesc Bootstrap 5 dark theme si `static/style.css` pentru stiluri comune.
Stilurile specifice fiecarei pagini raman in blocul `<style>` din fisierul respectiv.

Functionalitati comune in `index.html` si `compare.html`:
- **Spinner de incarcare** — overlay vizibil cat timp Flask proceseaza cererea
- **Autocomplete** — `<datalist>` populat din `/api/titles?q=` cu debounce 200ms

### Pipeline posteri TMDB

1. `_fetch_poster(tmdb_id)` verifica daca `static/posters/{tmdb_id}.jpg` exista local
2. Daca nu, apeleaza TMDB API v3 pentru `poster_path` proaspat
3. Descarca imaginea si o salveaza local
4. Flask serveste posterele din `static/posters/` — nu mai face request TMDB la urmatoarele cereri

TMDB API key: `0a6dca1e18d2d0c01590ab01163b14c3` (hardcodat in `main.py` la linia ~514)

## Output generat la rulare

| Fisier | Continut |
|--------|---------|
| `static/charts/data_exploration.png` | 3 grafice: distributie ratinguri, top genuri, filme/an |
| `static/charts/validation_results.png` | RMSE SVD vs. baseline, scatter real vs. prezis, Jaccard SVD vs. TF-IDF |

## Metrici model (valori tipice)

| Metrica | Valoare |
|---------|---------|
| RMSE SVD | 0.9372 |
| MAE SVD | 0.7252 |
| RMSE Baseline | 1.0460 |
| Imbunatatire | +10.4% |
| Jaccard SVD | 0.2358 |
| Jaccard TF-IDF | 0.4582 |
| Filme indexate | 9,025 |
| Utilizatori | 671 |
| Rating-uri | 99,810 |
