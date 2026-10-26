# -*- coding: utf-8 -*-
"""
Usage
-----
    PYTHONHASHSEED=0 python step_3_aligned_divergence.py

Outputs (./filtered_outputs/diachronic/aligned_divergence/{TIME_BINNING}/{SEED}/):
  - aligned_divergence.csv    per (term, period): aligned dist, unaligned dist,
                              frequency-matched floor + percentile, all-words
                              background median/p90/percentile, n_shared
  - background_divergences.csv  every shared word's aligned distance and
                              per-party count, per period (the raw material
                              behind both floors; consumed by background_noise.ipynb)
  - aligned_divergence_report.txt
  - aligned_divergence_small_multiples.png   one panel per term vs. noise floor
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: safe to run as a script
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases
from scipy.linalg import orthogonal_procrustes
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Word2Vec initialization seed. Change this ONE value and re-run to produce
# another replicate; outputs land in their own {SEED}/ directory so runs never
# overwrite each other.
SEED = 42

warnings.filterwarnings('ignore'); np.random.seed(SEED)

# ---------------------------------------------------------------- Configuration
TIME_BINNING = 'year'   # 'congress' (2-yr) gives more shared vocab => more stable
                        # alignment than 'year'. Switch to 'year' to match step_3.
DATA_PATH = './filtered_outputs/all_filtered_combined.csv'
OUTPUT_DIR = f'./filtered_outputs/diachronic/aligned_divergence/seed/{SEED}'
PERIOD_LABEL = 'Year' if TIME_BINNING == 'year' else 'Congress'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_DOCS_PER_PARTY_PERIOD = 10

K_FREQ_MATCH = 100

TARGET_TERMS = [
    'fraud', 'integrity', 'security', 'voter',
    'ballot', 'election', 'certification', 'audit',
    'stolen', 'rigged', 'access', 'mail',
    'illegal', 'legitimate', 'trust', 'transparency',
]

# workers=1 because gensim's multi-threaded async SGD is NOT reproducible even
# with a fixed seed. 50-dim skip-gram over 40 epochs: 200-dim CBOW collapsed on
# these small per-(party-)period corpora (all cosines ~0.998).
W2V_PARAMS = dict(vector_size=50, window=5, min_count=5, workers=1,
                  epochs=40, sg=1, negative=5, alpha=0.025, min_alpha=0.0001)

PALETTE = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c',
           '#e91e63', '#34495e', '#f1c40f', '#16a085', '#d35400', '#8e44ad',
           '#27ae60', '#2980b9', '#c0392b', '#7f8c8d', '#fd79a8']

# ------------------------------------------------------------------ Preprocess
stop_words = set(stopwords.words('english'))
stop_words.update({'would','could','should','may','might','must','said','says',
    'say','told','tell','get','got','getting','make','made','making','one','two',
    'three','first','second','last','also','however','therefore','thus','us','amp',
    'http','https','www'})

def preprocess(text):
    if pd.isna(text): return []
    return [t for t in word_tokenize(str(text).lower())
            if t.isalpha() and len(t) > 2 and t not in stop_words]

def l2norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


# ==============================================================================
# Figure
# ==============================================================================

def plot_small_multiples(div_df, save_path, ncols=4):
    """One small panel per term, each against its OWN frequency-matched floor.

    The dashed line is the median distance of the term's K_FREQ_MATCH nearest
    shared words in frequency, so it differs by term. A line riding above it =
    more cross-party divergent than comparably frequent words that year;
    on/below it = indistinguishable from them. The faint dotted line is the old
    all-words median, kept to show how far the frequency confound moved the
    floor. Panels share both axes; each title carries the term's mean
    frequency-matched percentile (p##). The shaded band is the gap to the
    matched floor (accent where above, gray where below).
    """
    all_floor = div_df.groupby('period')['bg_median'].first().sort_index()
    pct_by_term = div_df.groupby('term')['percentile_vs_freq_matched'].mean()
    # Alphabetical; keep only terms with >=2 points so a line can be drawn.
    order = sorted(t for t in pct_by_term.index if (div_df['term'] == t).sum() >= 2)
    nrows = int(np.ceil(len(order) / ncols))

    # Shared, data-driven limits so every panel is on one scale.
    ymin = float(min(div_df['divergence_aligned'].min(),
                     div_df['freq_matched_median'].min(), all_floor.min()))
    ymax = float(max(div_df['divergence_aligned'].max(),
                     div_df['freq_matched_median'].max(), all_floor.max()))
    pad = (ymax - ymin) * 0.08
    xmin, xmax = int(all_floor.index.min()), int(all_floor.index.max())
    xticks = sorted(set(range(xmin, xmax + 1, 5)) | {xmax})  # whole periods only

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.3),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, term in enumerate(order):
        ax = axes[i]
        td = div_df[div_df['term'] == term].sort_values('period')
        y = td['divergence_aligned'].values
        f_at = td['freq_matched_median'].values
        c = PALETTE[i % len(PALETTE)]

        ax.plot(all_floor.index, all_floor.values, ':', color='0.7', lw=1.0, zorder=1)
        ax.plot(td['period'], f_at, '--', color='0.4', lw=1.4, zorder=1)
        ax.plot(td['period'], y, marker='o', color=c, lw=2, markersize=4.5, zorder=3)
        ax.fill_between(td['period'], f_at, y, where=(y >= f_at), color=c,
                        alpha=0.18, zorder=2, interpolate=True)
        ax.fill_between(td['period'], f_at, y, where=(y < f_at), color='0.6',
                        alpha=0.18, zorder=2, interpolate=True)

        ax.set_title(f"{term}  (p{pct_by_term[term]:.0f})", fontsize=10,
                     fontweight='bold')
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xticks(xticks)
        ax.grid(True, alpha=0.25); ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    for ax in axes[len(order):]:
        ax.axis('off')

    # Bottom strip: legend only. The period label lives on the tick values
    # themselves; a supxlabel here collided with the legend box.
    fig.legend(handles=[Line2D([0], [0], color='0.4', ls='--', lw=1.4,
                               label=f'Frequency-matched floor (median of {K_FREQ_MATCH} '
                                     f'nearest-frequency words)'),
                        Line2D([0], [0], color='0.7', ls=':', lw=1.0,
                               label='All shared words (median; frequency-confounded)')],
               loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Aligned Partisan Divergence by Term vs. Frequency-Matched Floor',
                 fontsize=17, fontweight='bold')
    fig.supylabel('Partisan Divergence (Cosine Distance, aligned)',
                  fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.02, 0.045, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ------------------------------------------------------------------- Load data
print(f"Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
def year_to_congress(y):
    return None if pd.isna(y) or y < 2011 else 112 + (int(y) - 2011) // 2
df['Congress'] = df['Year'].apply(year_to_congress)
df['time_period'] = df['Congress'] if TIME_BINNING == 'congress' else df['Year']
df['Party'] = df['Party'].fillna('Unknown')
df = df[df['Party'].isin(['Democrat', 'Republican'])].copy()
df = df.dropna(subset=['time_period'])
df['time_period'] = df['time_period'].astype(int)

print("Tokenizing + phrasing ...")
df['tokens'] = df['Body_Subject'].apply(preprocess)
df = df[df['tokens'].str.len() >= 10].copy()
big = Phrases(df['tokens'].tolist(), min_count=5, threshold=10, delimiter='_')
tri = Phrases([big[t] for t in df['tokens']], min_count=5, threshold=10, delimiter='_')

PROTECT = set(TARGET_TERMS)
def phrase(tokens):
    out = []
    for tok in tri[big[tokens]]:
        parts = tok.split('_')
        out.extend(parts if any(p in PROTECT for p in parts) else [tok])
    return out
df['tf'] = df['tokens'].apply(phrase)

# --------------------------------------------- per-period train + align + score
rows = []
bg_rows = []   # full background distribution (every shared word's aligned divergence)
for period in sorted(df['time_period'].unique()):
    dem_docs = df[(df.time_period == period) & (df.Party == 'Democrat')]['tf'].tolist()
    rep_docs = df[(df.time_period == period) & (df.Party == 'Republican')]['tf'].tolist()
    if len(dem_docs) < MIN_DOCS_PER_PARTY_PERIOD or len(rep_docs) < MIN_DOCS_PER_PARTY_PERIOD:
        print(f"  {period}: Dem={len(dem_docs)} Rep={len(rep_docs)} - EXCLUDED")
        continue

    dem = Word2Vec(dem_docs, seed=SEED, **W2V_PARAMS)
    rep = Word2Vec(rep_docs, seed=SEED, **W2V_PARAMS)

    shared = sorted(set(dem.wv.index_to_key) & set(rep.wv.index_to_key))
    if len(shared) < 50:
        print(f"  {period}: only {len(shared)} shared words - SKIPPED")
        continue

    # Align rep -> dem the SAME way as by_party.py's align_models(): fit Procrustes
    # on the RAW shared-vocab vectors. Cosine distance is scale-invariant, so we
    # unit-normalize at DISTANCE time instead.
    dem_raw = np.array([dem.wv[w] for w in shared])
    rep_raw = np.array([rep.wv[w] for w in shared])
    R, _ = orthogonal_procrustes(rep_raw, dem_raw)      # rep_raw @ R ~= dem_raw

    # Background: post-alignment cosine distance for EVERY shared word = noise floor.
    bg = 1.0 - np.sum(l2norm(dem_raw) * l2norm(rep_raw @ R), axis=1)
    bg_median, bg_p90 = float(np.median(bg)), float(np.percentile(bg, 90))
    shared_idx = {w: i for i, w in enumerate(shared)}

    # Per-party training counts. The less-used party's vector is the noisier
    # one, so min(count_dem, count_rep) is what the distance's noise tracks.
    cnt_dem = np.array([dem.wv.get_vecattr(w, 'count') for w in shared])
    cnt_rep = np.array([rep.wv.get_vecattr(w, 'count') for w in shared])
    log_min_cnt = np.log(np.minimum(cnt_dem, cnt_rep))
    # Candidates for a frequency-matched set: every shared word except the
    # targets, so targets never set each other's floor.
    candidates = np.array([i for i, w in enumerate(shared) if w not in PROTECT])

    # Dump the full distribution so both floors are auditable: which words sit
    # where, and how frequency drives distance.
    bg_rows.extend({'period': period, 'word': w,
                    'divergence_aligned': round(float(bg[i]), 4),
                    'count_dem': int(cnt_dem[i]), 'count_rep': int(cnt_rep[i])}
                   for i, w in enumerate(shared))

    for term in TARGET_TERMS:
        # In both party vocabs => in `shared` => already scored in `bg`.
        if term not in shared_idx:
            continue
        i = shared_idx[term]
        dv = dem.wv[term] / (np.linalg.norm(dem.wv[term]) or 1)
        rv = rep.wv[term] / (np.linalg.norm(rep.wv[term]) or 1)
        d_aligned = float(bg[i])

        # K nearest non-target words in log frequency. Stable sort over the
        # (alphabetical) shared order breaks count ties deterministically.
        gap = np.abs(log_min_cnt[candidates] - log_min_cnt[i])
        matched = candidates[np.argsort(gap, kind='stable')[:K_FREQ_MATCH]]
        fm = bg[matched]
        fm_cnt = np.minimum(cnt_dem, cnt_rep)[matched]

        rows.append({'term': term, 'period': period,
                     'divergence_aligned': round(d_aligned, 4),
                     'divergence_unaligned_OLD': round(float(1 - np.dot(dv, rv)), 4),
                     'count_dem': int(cnt_dem[i]), 'count_rep': int(cnt_rep[i]),
                     'freq_matched_median': round(float(np.median(fm)), 4),
                     'freq_matched_p90': round(float(np.percentile(fm, 90)), 4),
                     'percentile_vs_freq_matched': round(float((fm < d_aligned).mean() * 100), 1),
                     'freq_matched_n': len(matched),
                     # The matched set's min-count range: a wide range around a
                     # top-frequency term means the match is loose.
                     'freq_matched_count_lo': int(fm_cnt.min()),
                     'freq_matched_count_hi': int(fm_cnt.max()),
                     'bg_median': round(bg_median, 4), 'bg_p90': round(bg_p90, 4),
                     'percentile_vs_background': round(float((bg < d_aligned).mean() * 100), 1),
                     'n_shared': len(shared)})

    print(f"  {period}: Dem={len(dem_docs)} Rep={len(rep_docs)} shared={len(shared)} "
          f"bg_median={bg_median:.3f}")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUTPUT_DIR, 'aligned_divergence.csv'), index=False)
pd.DataFrame(bg_rows).to_csv(os.path.join(OUTPUT_DIR, 'background_divergences.csv'),
                             index=False)

# ------------------------------------------------------------------- Summarize
lines = ["=" * 78,
         "VALID CROSS-PARTY DIVERGENCE (within-period Dem<->Rep Procrustes alignment)",
         "=" * 78,
         f"Time binning: {TIME_BINNING}   Periods scored: {out['period'].nunique()}",
         "",
         "How to read: 'divergence_aligned' is cosine distance AFTER aligning the two",
         "party frames. 'fm_pctile' is where the term sits among the "
         f"{K_FREQ_MATCH} shared words",
         "closest to it in frequency (50 = typical for a word that common; 90+ = more",
         "divergent than most comparably frequent words). A RELATIVE floor, not a null:",
         "for divergence beyond chance see step_3_cosine_baseline.py (perm arm).",
         "'all_pctile' ranks against ALL shared words -- frequency-confounded, shown",
         "for contrast only. OLD = unaligned (rotation-noise) distance.",
         ""]

# Per-term: mean aligned divergence + mean frequency-matched percentile (the headline)
g = out.groupby('term').agg(mean_aligned=('divergence_aligned', 'mean'),
                            mean_fm=('freq_matched_median', 'mean'),
                            mean_pctile=('percentile_vs_freq_matched', 'mean'),
                            mean_all=('percentile_vs_background', 'mean'),
                            mean_old=('divergence_unaligned_OLD', 'mean'),
                            n=('period', 'nunique')).sort_values('mean_pctile', ascending=False)
lines.append("PER-TERM (sorted by divergence vs. comparably frequent words):")
lines.append(f"  {'term':16}{'aligned':>9}{'fm_floor':>10}{'fm_pctile':>11}"
             f"{'all_pctile':>12}{'OLD':>8}{'n':>4}")
for term, r in g.iterrows():
    lines.append(f"  {term:16}{r['mean_aligned']:>9.3f}{r['mean_fm']:>10.3f}"
                 f"{r['mean_pctile']:>10.0f}%{r['mean_all']:>11.0f}%"
                 f"{r['mean_old']:>8.3f}{int(r['n']):>4}")

# Temporal trend of how divergent the target terms are vs their matched floors
per = out.groupby('period').agg(mean_pctile=('percentile_vs_freq_matched', 'mean'),
                                mean_aligned=('divergence_aligned', 'mean')).sort_index()
lines += ["", "TEMPORAL TREND (mean target-term frequency-matched percentile, by period):"]
for p, r in per.iterrows():
    lines.append(f"  {p}: pctile={r['mean_pctile']:.0f}%   aligned_dist={r['mean_aligned']:.3f}")
if len(per) >= 4:
    half = len(per) // 2
    early = per['mean_pctile'].iloc[:half].mean()
    late = per['mean_pctile'].iloc[half:].mean()
    lines += ["", f"Early periods mean percentile: {early:.0f}%",
              f"Late periods mean percentile:  {late:.0f}%",
              f"Change: {late - early:+.0f} pts  "
              f"({'INCREASING' if late > early else 'DECREASING/FLAT'})",
              "", "INTERPRETATION:",
              "  If late >> early and percentiles are high (80+), there is a real, growing",
              "  cross-party divergence once rotation is removed. If percentiles hover near",
              "  50 or the trend is flat, the target terms are no more divergent than words",
              "  of similar frequency -> no quantitative divergence story; lean on",
              "  qualitative neighbors."]

report = "\n".join(lines)
print("\n" + report)
with open(os.path.join(OUTPUT_DIR, 'aligned_divergence_report.txt'), 'w') as f:
    f.write(report + "\n")

plot_small_multiples(out, os.path.join(OUTPUT_DIR, 'aligned_divergence_small_multiples.png'))
print(f"\nOutputs -> {OUTPUT_DIR}/")
