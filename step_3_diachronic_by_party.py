# -*- coding: utf-8 -*-
"""
Usage:
    PYTHONHASHSEED=0 python step_3_diachronic_by_party.py

"""

# !pip install gensim matplotlib seaborn scipy scikit-learn pandas numpy -q

# Import libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: don't block on plt.show() when run as a script
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from scipy.linalg import orthogonal_procrustes
import warnings
import os
from tqdm import tqdm

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set style
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 8)
warnings.filterwarnings('ignore')

"""# Configuration"""
# Word2Vec initialization seed. Change this ONE value and re-run to produce
# another replicate; outputs land in their own {SEED}/ directory so runs never
# overwrite each other. Used by BOTH Word2Vec calls below (Dem and Rep) —
# varying only one of them would compare a re-initialized party model against a
# fixed reference. (Same convention as step_3_aligned_divergence.py.)
SEED = 42

np.random.seed(SEED)

# Time binning
TIME_BINNING = 'year'  # Options: 'congress', 'year'  (2-yr bins: ~2x docs/cell)

# Paths — seed-scoped, so replicates sit side by side instead of clobbering
# each other. Downstream consumers (step_3_overlap_baseline.py, the two
# step_4_* scripts) point at a specific seed directory.
DATA_PATH = './filtered_outputs/all_filtered_combined.csv'
OUTPUT_DIR = f'./filtered_outputs/diachronic/by_party/{TIME_BINNING}/{SEED}'
FIG_DIR = f'./filtered_outputs/diachronic/by_party/figures/{TIME_BINNING}/{SEED}'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Target terms to track. Same 16 as aligned_divergence. This list is also the
# phrasing `protect` set (see Text Preprocessing), so editing it changes the
# token stream the models train on, not just what gets reported. Must stay the
# same as TARGET_TERMS in step_3_aligned_divergence.py and
# step_3_overlap_baseline.py.
#
# Dropped from v1: `election_integrity` (unmerged into `election` + `integrity`
# by the protected phrasing, so it no longer exists as a token), and
# `suppression`, `machines`, `absentee`, `verify` (counts reshaped by the
# phrasing change).
TARGET_TERMS = [
    'fraud', 'integrity', 'security', 'voter',
    'ballot', 'election', 'certification', 'audit',
    'stolen', 'rigged', 'access', 'mail',
    'illegal', 'legitimate', 'trust', 'transparency',
]

# Word2Vec hyperparameters. Changed from v1 (200-dim CBOW, 10 epochs), which
# collapsed on these small per-party-year corpora (all cosines ~0.998): 50-dim
# skip-gram over 40 epochs. workers=1 because gensim's multi-threaded SGD is not
# reproducible even with a fixed seed.
W2V_PARAMS = {
    'vector_size': 50,
    'window': 5,
    'min_count': 5,          # Lower threshold since splitting by party
    'workers': 1,
    'epochs': 40,
    'sg': 1,
    'negative': 5,
    'alpha': 0.025,
    'min_alpha': 0.0001
}

# Minimum documents per party per period
MIN_DOCS_PER_PARTY_PERIOD = 10  # Lower for yearly bins

# Colors for parties
PARTY_COLORS = {
    'Democrat': '#3498db',      # Blue
    'Republican': '#e74c3c',    # Red
    'combined': '#95a5a6'       # Gray
}

print(f"Word2Vec seed: {SEED}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Tracking {len(TARGET_TERMS)} terms")
print(f"Minimum {MIN_DOCS_PER_PARTY_PERIOD} docs per party per period")

"""#Load and Prepare Data"""

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} documents")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

# Ensure Congress column
if 'Congress' not in df.columns or df['Congress'].isna().any():
    def year_to_congress(year):
        if pd.isna(year) or year < 2011:
            return None
        return 112 + (int(year) - 2011) // 2
    df['Congress'] = df['Year'].apply(year_to_congress)

# Select time period
if TIME_BINNING == 'congress':
    df['time_period'] = df['Congress']
    period_label = 'Congress'
elif TIME_BINNING == 'year':
    df['time_period'] = df['Year']
    period_label = 'Year'

# Clean party labels
df['Party'] = df['Party'].fillna('Unknown')
df = df[df['Party'].isin(['Democrat', 'Republican'])].copy()

print(f"\nAfter filtering to Dem/Rep: {len(df):,} documents")

# Check distribution
print("\nDocuments by party:")
print(df['Party'].value_counts())

print("\nDocuments by period and party:")
print(df.groupby(['time_period', 'Party']).size().unstack(fill_value=0))

# Filter periods with sufficient data for BOTH parties
valid_periods = []
for period in df['time_period'].unique():
    dem_count = len(df[(df['time_period'] == period) & (df['Party'] == 'Democrat')])
    rep_count = len(df[(df['time_period'] == period) & (df['Party'] == 'Republican')])

    if dem_count >= MIN_DOCS_PER_PARTY_PERIOD and rep_count >= MIN_DOCS_PER_PARTY_PERIOD:
        valid_periods.append(period)
        print(f"{period}: Dem={dem_count}, Rep={rep_count}")
    else:
        print(f"{period}: Dem={dem_count}, Rep={rep_count} - EXCLUDED")

df = df[df['time_period'].isin(valid_periods)].copy()

print(f"Valid periods: {sorted(valid_periods)}")
print(f"Final corpus: {len(df):,} documents")

"""#Text Preprocessing (Same for Both Parties)"""

# Load stopwords
stop_words = set(stopwords.words('english'))
custom_stopwords = {
    'would', 'could', 'should', 'may', 'might', 'must',
    'said', 'says', 'say', 'told', 'tell',
    'get', 'got', 'getting', 'make', 'made', 'making',
    'one', 'two', 'three', 'first', 'second', 'last',
    'also', 'however', 'therefore', 'thus',
    'us', 'amp', 'http', 'https', 'www'
}
stop_words.update(custom_stopwords)

def preprocess_text(text):
    """Tokenize and clean text"""
    if pd.isna(text):
        return []

    text = str(text).lower()
    tokens = word_tokenize(text)

    tokens = [
        token for token in tokens
        if (token.isalpha() and len(token) > 2 and token not in stop_words)
    ]

    return tokens

# Preprocess
tqdm.pandas(desc="Tokenizing")
df['tokens'] = df['Body_Subject'].progress_apply(preprocess_text)
df = df[df['tokens'].str.len() >= 10].copy()

print(f"\n Preprocessing complete")
print(f"Documents retained: {len(df):,}")
print(f"Average tokens: {df['tokens'].str.len().mean():.1f}")

# Build phrase models on full corpus (to ensure consistency)
print("\nBuilding phrase models on full corpus...")
all_tokens = df['tokens'].tolist()

bigram = Phrases(all_tokens, min_count=5, threshold=10, delimiter='_')
df['tokens_bigram'] = df['tokens'].apply(lambda x: bigram[x])

bigram_tokens = df['tokens_bigram'].tolist()
trigram = Phrases(bigram_tokens, min_count=5, threshold=10, delimiter='_')

# New in v2: the phraser greedily rewrites "voter fraud" -> voter_fraud, and
# each merge consumes one standalone `voter`, draining it below min_count in
# sparse party-years. So undo any merge that swallows a target term; the
# absorbed context reappears as neighbors instead. This also splits
# `election_integrity` back into `election` + `integrity`.
PROTECT = set(TARGET_TERMS)
def unmerge_targets(tokens):
    out = []
    for tok in tokens:
        parts = tok.split('_')
        out.extend(parts if any(p in PROTECT for p in parts) else [tok])
    return out

df['tokens_final'] = df['tokens_bigram'].apply(lambda x: unmerge_targets(trigram[x]))



"""#Train Party-Specific Models by Time Period"""

# Storage
dem_models = {}
rep_models = {}
dem_vocab = {}
rep_vocab = {}
corpus_stats = []

for period in sorted(df['time_period'].unique()):
    print(f"\n{'='*80}")
    print(f"{period_label} {period}")
    print('='*80)

    # Democrat corpus
    dem_docs = df[
        (df['time_period'] == period) &
        (df['Party'] == 'Democrat')
    ]['tokens_final'].tolist()

    # Republican corpus
    rep_docs = df[
        (df['time_period'] == period) &
        (df['Party'] == 'Republican')
    ]['tokens_final'].tolist()

    print(f"Democrat documents: {len(dem_docs):,}")
    print(f"Republican documents: {len(rep_docs):,}")

    # Train Democrat model
    print("Training Democrat model...")
    dem_model = Word2Vec(
        sentences=dem_docs,
        **W2V_PARAMS,
        seed=SEED
    )
    dem_models[period] = dem_model
    dem_vocab[period] = set(dem_model.wv.index_to_key)
    print(f"  Vocabulary: {len(dem_model.wv):,} words")

    # Save
    dem_path = os.path.join(OUTPUT_DIR, f'dem_model_{period}.bin')
    dem_model.save(dem_path)

    # Train Republican model
    print("Training Republican model...")
    rep_model = Word2Vec(
        sentences=rep_docs,
        **W2V_PARAMS,
        seed=SEED
    )
    rep_models[period] = rep_model
    rep_vocab[period] = set(rep_model.wv.index_to_key)
    print(f"  Vocabulary: {len(rep_model.wv):,} words")

    # Save
    rep_path = os.path.join(OUTPUT_DIR, f'rep_model_{period}.bin')
    rep_model.save(rep_path)

    # Track stats
    corpus_stats.append({
        'period': period,
        'dem_docs': len(dem_docs),
        'rep_docs': len(rep_docs),
        'dem_vocab': len(dem_vocab[period]),
        'rep_vocab': len(rep_vocab[period]),
        'shared_vocab': len(dem_vocab[period].intersection(rep_vocab[period]))
    })

print(f"\n Trained {len(dem_models)} Democrat models")
print(f" Trained {len(rep_models)} Republican models")

# Save stats
stats_df = pd.DataFrame(corpus_stats)
stats_path = os.path.join(OUTPUT_DIR, 'corpus_statistics.csv')
stats_df.to_csv(stats_path, index=False)

"""#Align Democrat Models Over Time"""

# Find shared Democrat vocabulary across all periods
dem_anchor = sorted(dem_models.keys())[0]
dem_shared_vocab = dem_vocab[dem_anchor].copy()

for period in dem_models.keys():
    dem_shared_vocab = dem_shared_vocab.intersection(dem_vocab[period])

dem_shared_vocab_list = sorted(list(dem_shared_vocab))

print(f"Anchor period: {dem_anchor}")
print(f"Democrat shared vocabulary: {len(dem_shared_vocab_list):,} words")

def align_models(source_model, target_model, shared_vocab):
    """Align source model to target using Orthogonal Procrustes"""
    source_vecs = np.array([source_model.wv[word] for word in shared_vocab])
    target_vecs = np.array([target_model.wv[word] for word in shared_vocab])

    rotation_matrix, _ = orthogonal_procrustes(source_vecs, target_vecs)
    aligned_vecs = source_model.wv.vectors @ rotation_matrix

    return aligned_vecs, rotation_matrix

# Align Democrat models
dem_aligned = {}
dem_aligned[dem_anchor] = dem_models[dem_anchor].wv.vectors

for period in sorted(dem_models.keys()):
    if period == dem_anchor:
        continue

    print(f"Aligning Democrat {period} to {dem_anchor}...")
    aligned_vecs, rotation = align_models(
        dem_models[period],
        dem_models[dem_anchor],
        dem_shared_vocab_list
    )
    dem_aligned[period] = aligned_vecs

"""#Align Republican Models Over Time"""

# Find shared Republican vocabulary
rep_anchor = sorted(rep_models.keys())[0]
rep_shared_vocab = rep_vocab[rep_anchor].copy()

for period in rep_models.keys():
    rep_shared_vocab = rep_shared_vocab.intersection(rep_vocab[period])

rep_shared_vocab_list = sorted(list(rep_shared_vocab))

print(f"Anchor period: {rep_anchor}")
print(f"Republican shared vocabulary: {len(rep_shared_vocab_list):,} words")

# Align Republican models
rep_aligned = {}
rep_aligned[rep_anchor] = rep_models[rep_anchor].wv.vectors

for period in sorted(rep_models.keys()):
    if period == rep_anchor:
        continue

    print(f"Aligning Republican {period} to {rep_anchor}...")
    aligned_vecs, rotation = align_models(
        rep_models[period],
        rep_models[rep_anchor],
        rep_shared_vocab_list
    )
    rep_aligned[period] = aligned_vecs

"""#Track Partisan Neighbors Over Time"""

def get_neighbors(model, word, k=25, aligned_vecs=None):
    """Get k nearest neighbors"""
    if word not in model.wv:
        return []

    word_idx = model.wv.key_to_index[word]

    if aligned_vecs is not None:
        word_vec = aligned_vecs[word_idx]

        similarities = []
        for i, other_word in enumerate(model.wv.index_to_key):
            if other_word == word:
                continue
            other_vec = aligned_vecs[i]
            sim = 1 - cosine(word_vec, other_vec)
            similarities.append((other_word, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    else:
        return model.wv.most_similar(word, topn=k)

# Track for each term and party
partisan_neighbors = {
    'Democrat': {},
    'Republican': {}
}

for term in TARGET_TERMS:
    print(f"\n{'='*80}")
    print(f"{term.upper()}")
    print('='*80)

    partisan_neighbors['Democrat'][term] = {}
    partisan_neighbors['Republican'][term] = {}

    for period in sorted(dem_models.keys()):
        print(f"\n{period}:")

        # Democrat neighbors
        if term in dem_vocab[period]:
            dem_neighbors = get_neighbors(
                dem_models[period],
                term,
                k=25,
                aligned_vecs=dem_aligned[period]
            )
            partisan_neighbors['Democrat'][term][period] = dem_neighbors
            dem_str = ", ".join([f"{w}({s:.2f})" for w, s in dem_neighbors[:5]])
            print(f"  Democrat:   {dem_str}")
        else:
            print(f"  Democrat:   [not in vocabulary]")

        # Republican neighbors
        if term in rep_vocab[period]:
            rep_neighbors = get_neighbors(
                rep_models[period],
                term,
                k=25,  # match the Democrat call (was 10) so the two party panels are symmetric
                aligned_vecs=rep_aligned[period]
            )
            partisan_neighbors['Republican'][term][period] = rep_neighbors
            rep_str = ", ".join([f"{w}({s:.2f})" for w, s in rep_neighbors[:5]])
            print(f"  Republican: {rep_str}")
        else:
            print(f"  Republican: [not in vocabulary]")

# Save neighbors in long/tidy format for downstream network visualization (step 4)
partisan_rows = []
for party in ['Democrat', 'Republican']:
    for term in TARGET_TERMS:
        for period, neighbors in partisan_neighbors[party].get(term, {}).items():
            for rank, (word, score) in enumerate(neighbors, start=1):
                partisan_rows.append({
                    'term': term,
                    'year': period,
                    'party': party,
                    'neighbor': word,
                    'similarity_score': score,
                    'rank': rank
                })

partisan_csv_path = os.path.join(OUTPUT_DIR, 'neighbors_partisan.csv')
pd.DataFrame(partisan_rows).to_csv(partisan_csv_path, index=False)
print(f"✓ Partisan neighbors saved to: {partisan_csv_path}")

# Save to file
neighbors_path = os.path.join(OUTPUT_DIR, 'partisan_neighbors_over_time.txt')
with open(neighbors_path, 'w') as f:
    for term in TARGET_TERMS:
        f.write(f"\n{'='*80}\n")
        f.write(f"{term.upper()}\n")
        f.write('='*80 + '\n')

        for period in sorted(dem_models.keys()):
            f.write(f"\n{period}:\n")

            # Democrat
            if period in partisan_neighbors['Democrat'][term]:
                neighbors = partisan_neighbors['Democrat'][term][period]
                neighbor_str = ", ".join([f"{w}({s:.2f})" for w, s in neighbors])
                f.write(f"  Democrat:   {neighbor_str}\n")
            else:
                f.write(f"  Democrat:   [not in vocabulary]\n")

            # Republican
            if period in partisan_neighbors['Republican'][term]:
                neighbors = partisan_neighbors['Republican'][term][period]
                neighbor_str = ", ".join([f"{w}({s:.2f})" for w, s in neighbors])
                f.write(f"  Republican: {neighbor_str}\n")
            else:
                f.write(f"  Republican: [not in vocabulary]\n")

"""#Calculate Within-Party Semantic Shifts"""

def calculate_shift(model1, model2, word, aligned_vecs1, aligned_vecs2):
    """Calculate shift between two time periods"""
    if word not in model1.wv or word not in model2.wv:
        return None

    idx1 = model1.wv.key_to_index[word]
    idx2 = model2.wv.key_to_index[word]

    vec1 = aligned_vecs1[idx1]
    vec2 = aligned_vecs2[idx2]

    return cosine(vec1, vec2)

# Calculate shifts
dem_shifts = []
rep_shifts = []

periods = sorted(dem_models.keys())

for i in range(len(periods) - 1):
    period1 = periods[i]
    period2 = periods[i + 1]

    for term in TARGET_TERMS:
        # Democrat shift
        dem_shift = calculate_shift(
            dem_models[period1],
            dem_models[period2],
            term,
            dem_aligned[period1],
            dem_aligned[period2]
        )

        if dem_shift is not None:
            dem_shifts.append({
                'party': 'Democrat',
                'term': term,
                'period1': period1,
                'period2': period2,
                'shift': dem_shift
            })

        # Republican shift
        rep_shift = calculate_shift(
            rep_models[period1],
            rep_models[period2],
            term,
            rep_aligned[period1],
            rep_aligned[period2]
        )

        if rep_shift is not None:
            rep_shifts.append({
                'party': 'Republican',
                'term': term,
                'period1': period1,
                'period2': period2,
                'shift': rep_shift
            })

# Combine
all_shifts = pd.DataFrame(dem_shifts + rep_shifts)

# Save
shifts_path = os.path.join(OUTPUT_DIR, 'partisan_semantic_shifts.csv')
all_shifts.to_csv(shifts_path, index=False)

print(f"\nTotal shift measurements: {len(all_shifts)}")

# Show top shifts by party
print("\nLargest Democrat shifts:")
print(all_shifts[all_shifts['party'] == 'Democrat'].nlargest(10, 'shift')[
    ['term', 'period1', 'period2', 'shift']
])

print("\nLargest Republican shifts:")
print(all_shifts[all_shifts['party'] == 'Republican'].nlargest(10, 'shift')[
    ['term', 'period1', 'period2', 'shift']
])

"""#Asymmetric Polarization (mean within-party shift by party)"""
# all_shifts = one row per (party, term, period-transition). Mean shift per
# party = how much that party's vocabulary moved overall; the ratio is the
# "one party shifted X% more" statistic (the draft's 46% was computed by hand
# from this CSV and shipped without a CI). Bootstrapped here so it ships with one.
# NOTE: pools across terms AND transitions, so it's a coarse summary; the CI
# tells you whether the gap is distinguishable from zero.
party_means = all_shifts.groupby('party')['shift'].mean()
dem_mean = party_means.get('Democrat', float('nan'))
rep_mean = party_means.get('Republican', float('nan'))
hi, lo = max(dem_mean, rep_mean), min(dem_mean, rep_mean)
pct_diff = (hi - lo) / lo * 100 if lo else float('nan')

# Bootstrap RESAMPLING seed — deliberately NOT tied to SEED. SEED controls
# Word2Vec initialization; this one only controls which rows the bootstrap draws
# from an already-computed all_shifts table. Holding it fixed across replicates
# is common-random-numbers: the CI width then varies only because the underlying
# shifts changed, not because the resampling draws changed too.
rng = np.random.RandomState(42)
dem_sh = all_shifts.loc[all_shifts['party'] == 'Democrat', 'shift'].values
rep_sh = all_shifts.loc[all_shifts['party'] == 'Republican', 'shift'].values
boot = []
for _ in range(2000):
    d = rng.choice(dem_sh, size=len(dem_sh), replace=True).mean()
    r = rng.choice(rep_sh, size=len(rep_sh), replace=True).mean()
    h, l = max(d, r), min(d, r)
    boot.append((h - l) / l * 100 if l else np.nan)
ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5])

larger = 'Republican' if rep_mean > dem_mean else 'Democrat'
print(f"\nASYMMETRIC POLARIZATION")
print(f"  Democrat mean shift:   {dem_mean:.3f}  (n={len(dem_sh)})")
print(f"  Republican mean shift: {rep_mean:.3f}  (n={len(rep_sh)})")
print(f"  {larger} shifted {pct_diff:.0f}% more (95% CI: {ci_lo:.0f}%-{ci_hi:.0f}%)")

pd.DataFrame([{
    'dem_mean_shift': dem_mean, 'rep_mean_shift': rep_mean,
    'larger_party': larger, 'pct_difference': pct_diff,
    'ci95_low': ci_lo, 'ci95_high': ci_hi,
    'n_dem': len(dem_sh), 'n_rep': len(rep_sh),
}]).to_csv(os.path.join(OUTPUT_DIR, 'partisan_shift_asymmetry.csv'), index=False)

"""#Calculate Cross-Party Divergence"""
# Ported back from the original step_3_diachronic_by_party.py (dropped in v2).
# NOTE (unchanged from original): dem_aligned and rep_aligned live in SEPARATELY
# anchored spaces (each party aligned to its own earliest period), so this cosine
# distance is an approximation — it tracks relative divergence trends over time,
# not an absolute cross-party distance. This is the noisy plot we discussed;
# reproduced here exactly as it was originally computed and plotted.

def calculate_partisan_divergence(term, dem_model, rep_model, dem_aligned_vecs, rep_aligned_vecs):
    """
    Calculate divergence between Democrat and Republican embeddings

    Note: This requires aligning Dem and Rep spaces to each other
    For simplicity, we use cosine distance in their respective aligned spaces
    """
    if term not in dem_model.wv or term not in rep_model.wv:
        return None

    dem_idx = dem_model.wv.key_to_index[term]
    rep_idx = rep_model.wv.key_to_index[term]

    dem_vec = dem_aligned_vecs[dem_idx]
    rep_vec = rep_aligned_vecs[rep_idx]

    # Note: These vectors are in different spaces, so this is an approximation
    # For true cross-party comparison, we'd need to align Dem→Rep or vice versa
    # But for relative comparison across time, this captures divergence trends
    return cosine(dem_vec, rep_vec)

# Calculate divergence for each term in each period
divergence_data = []

for period in sorted(dem_models.keys()):
    for term in TARGET_TERMS:
        div = calculate_partisan_divergence(
            term,
            dem_models[period],
            rep_models[period],
            dem_aligned[period],
            rep_aligned[period]
        )

        if div is not None:
            divergence_data.append({
                'period': period,
                'term': term,
                'divergence': div
            })

divergence_df = pd.DataFrame(divergence_data)

# Save
div_path = os.path.join(OUTPUT_DIR, 'partisan_divergence.csv')
divergence_df.to_csv(div_path, index=False)

print(f"\nTotal divergence measurements: {len(divergence_df)}")

# Show terms with highest average divergence
print("\nTerms with highest average divergence:")
avg_div = divergence_df.groupby('term')['divergence'].mean().sort_values(ascending=False)
print(avg_div.head(15))

# Show periods with highest average divergence
print("\nPeriods with highest average divergence:")
period_div = divergence_df.groupby('period')['divergence'].mean().sort_values(ascending=False)
print(period_div)

"""# Visualization - Partisan Divergence Timeline"""

def plot_divergence_timeline(divergence_df, terms=None, save_path=FIG_DIR):
    """Line plot of partisan divergence over time"""
    if terms is None:
        # Select terms with most data
        term_counts = divergence_df['term'].value_counts()
        terms = term_counts.head(17).index.tolist()

    # Define distinct color palette for up to 17 terms
    colors = [
        '#e74c3c',  # Red
        '#3498db',  # Blue
        '#2ecc71',  # Green
        '#f39c12',  # Orange
        '#9b59b6',  # Purple
        '#1abc9c',  # Turquoise
        '#e91e63',  # Pink
        '#34495e',  # Dark Gray
        '#f1c40f',  # Yellow
        '#16a085',  # Dark Turquoise
        '#d35400',  # Brown
        '#8e44ad',  # Deep Violet
        '#27ae60',  # Forest Green
        '#2980b9',  # Steel Blue
        '#c0392b',  # Dark Red
        '#7f8c8d',  # Medium Gray
        '#fd79a8',  # Soft Rose
    ]

    # Define distinct marker styles
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P']

    fig, ax = plt.subplots(figsize=(14, 7))

    for idx, term in enumerate(terms):
        term_data = divergence_df[divergence_df['term'] == term].sort_values('period')

        if len(term_data) < 2:
            continue

        ax.plot(
            term_data['period'],
            term_data['divergence'],
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=2.5,
            markersize=10,
            label=term,
            alpha=0.85
        )

    ax.set_xlabel(period_label, fontsize=14, fontweight='bold')
    ax.set_ylabel('Partisan Divergence (Cosine Distance)', fontsize=14, fontweight='bold')
    ax.set_title('Partisan Divergence Over Time', fontsize=18, fontweight='bold')
    ax.legend(title='Term', fontsize=11, title_fontsize=12, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()

save_path = os.path.join(FIG_DIR, 'divergence_timeline.png')
plot_divergence_timeline(divergence_df, save_path=save_path)

"""#Visualization - Side-by-Side Neighbor Comparison"""

def plot_partisan_neighbor_comparison(term, partisan_neighbors, periods, save_path=FIG_DIR):
    """
    Create side-by-side heatmaps showing Dem vs Rep neighbor evolution
    """
    # Get Democrat neighbors
    dem_neighbors_dict = partisan_neighbors['Democrat'].get(term, {})
    rep_neighbors_dict = partisan_neighbors['Republican'].get(term, {})

    if not dem_neighbors_dict or not rep_neighbors_dict:
        print(f"  Insufficient data for {term}")
        return

    # Collect all unique neighbors from both parties
    all_neighbors = set()
    for period_neighbors in dem_neighbors_dict.values():
        all_neighbors.update([w for w, s in period_neighbors])
    for period_neighbors in rep_neighbors_dict.values():
        all_neighbors.update([w for w, s in period_neighbors])

    all_neighbors = sorted(list(all_neighbors))
    valid_periods = sorted([p for p in periods if p in dem_neighbors_dict and p in rep_neighbors_dict])

    if len(valid_periods) == 0 or len(all_neighbors) == 0:
        print(f"  No valid data for {term}")
        return

    # Create matrices
    dem_matrix = np.zeros((len(all_neighbors), len(valid_periods)))
    rep_matrix = np.zeros((len(all_neighbors), len(valid_periods)))

    for j, period in enumerate(valid_periods):
        # Democrat
        dem_period_neighbors = {w: s for w, s in dem_neighbors_dict[period]}
        for i, neighbor in enumerate(all_neighbors):
            if neighbor in dem_period_neighbors:
                dem_matrix[i, j] = dem_period_neighbors[neighbor]

        # Republican
        rep_period_neighbors = {w: s for w, s in rep_neighbors_dict[period]}
        for i, neighbor in enumerate(all_neighbors):
            if neighbor in rep_period_neighbors:
                rep_matrix[i, j] = rep_period_neighbors[neighbor]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, max(10, len(all_neighbors) * 0.25)))

    # Democrat heatmap
    sns.heatmap(
        dem_matrix,
        xticklabels=valid_periods,
        yticklabels=all_neighbors,
        cmap='Blues',
        cbar_kws={'label': 'Similarity'},
        vmin=0,
        vmax=1,
        ax=axes[0]
    )
    axes[0].set_title(f'Democrat: "{term.replace("_", " ")}"', fontsize=14, fontweight='bold', color=PARTY_COLORS['Democrat'])
    axes[0].set_xlabel(period_label, fontsize=11)

    # Republican heatmap
    sns.heatmap(
        rep_matrix,
        xticklabels=valid_periods,
        yticklabels=all_neighbors,
        cmap='Reds',
        cbar_kws={'label': 'Similarity'},
        vmin=0,
        vmax=1,
        ax=axes[1]
    )
    axes[1].set_title(f'Republican: "{term.replace("_", " ")}"', fontsize=14, fontweight='bold', color=PARTY_COLORS['Republican'])
    axes[1].set_xlabel(period_label, fontsize=11)

    plt.suptitle(f'Partisan Neighbor Evolution: "{term.replace("_", " ")}"', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()

# Plot for key terms
key_terms = ['integrity', 'fraud', 'security', 'voter', 'ballot', 'election']

for term in key_terms:
    print(f"\nPlotting partisan neighbor comparison: {term}")
    save_path = os.path.join(FIG_DIR, f'neighbor_comparison_{term}.png')
    plot_partisan_neighbor_comparison(term, partisan_neighbors, periods, save_path)

"""#Visualization - Within-Party Shift Timelines"""

def plot_party_shift_timeline(shift_df, party, terms=None, save_path=FIG_DIR):
    """
    Line plot showing semantic shift magnitude over time for ONE party
    Similar to combined analysis but party-specific
    """
    party_data = shift_df[shift_df['party'] == party].copy()

    if terms is None:
        # Select terms with most complete data
        term_counts = party_data['term'].value_counts()
        terms = term_counts.head(6).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 7))

    # Build one shared, ordered x-axis from every transition across the selected
    # terms, so each term's points land on the same coordinate (different terms
    # cover different periods). Plotting against each term's own row index
    # misaligns the lines and lets longer series run past the labels.
    plot_data = party_data[party_data['term'].isin(terms)].copy()
    transitions = plot_data[['period1', 'period2']].drop_duplicates().sort_values('period1')
    transitions['transition'] = transitions['period1'].astype(str) + '→' + transitions['period2'].astype(str)
    x_labels = transitions['transition'].tolist()
    x_pos = {t: i for i, t in enumerate(x_labels)}

    for term in terms:
        term_data = party_data[party_data['term'] == term].copy()

        if len(term_data) < 2:
            continue

        term_data = term_data.sort_values('period1')
        term_data['transition'] = term_data['period1'].astype(str) + '→' + term_data['period2'].astype(str)

        ax.plot(
            [x_pos[t] for t in term_data['transition']],
            term_data['shift'],
            marker='o',
            linewidth=2.5,
            markersize=10,
            label=term,
            alpha=0.8
        )

    # Set x-axis labels
    if len(x_labels) > 0:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

    ax.set_xlabel('Period Transition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Semantic Shift (Cosine Distance)', fontsize=12, fontweight='bold')

    party_color = PARTY_COLORS.get(party, 'gray')
    ax.set_title(f'{party} Semantic Shift Over Time',
                 fontsize=18, fontweight='bold', color=party_color)

    ax.legend(title='Term', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()

# Plot Democrat shift timeline
print("\nPlotting Democrat shift timeline")
dem_save_path = os.path.join(FIG_DIR, 'democrat_shift_timeline.png')
plot_party_shift_timeline(
    all_shifts,
    'Democrat',
    terms=['integrity', 'fraud', 'security', 'voter', 'ballot', 'stolen'],
    save_path=dem_save_path
)

# Plot Republican shift timeline
print("\nPlotting Republican shift timeline")
rep_save_path = os.path.join(FIG_DIR, 'republican_shift_timeline.png')
plot_party_shift_timeline(
    all_shifts,
    'Republican',
    terms=['integrity', 'fraud', 'security', 'voter', 'ballot', 'stolen', 'election'],
    save_path=rep_save_path
)

# Create combined comparison plot
print("\nCreating side-by-side party comparison")

def plot_comparative_shift_timeline(shift_df, term, save_path=FIG_DIR):
    """
    Show Democrat and Republican shifts for the same term side-by-side
    """
    dem_data = shift_df[(shift_df['party'] == 'Democrat') & (shift_df['term'] == term)].sort_values('period1')
    rep_data = shift_df[(shift_df['party'] == 'Republican') & (shift_df['term'] == term)].sort_values('period1')

    if len(dem_data) == 0 or len(rep_data) == 0:
        print(f"  Insufficient data for {term}")
        return

    # Build ONE shared, ordered transition axis from the union of both parties'
    # transitions. Each party is plotted against this common axis (missing
    # transitions -> NaN, so the line breaks at a gap instead of compressing).
    # This keeps Dem and Rep aligned period-for-period and comparable.
    for d in (dem_data, rep_data):
        d['transition'] = d['period1'].astype(str) + '→' + d['period2'].astype(str)

    transitions = (
        pd.concat([dem_data[['period1', 'transition']], rep_data[['period1', 'transition']]])
        .drop_duplicates('transition')
        .sort_values('period1')
    )
    x_labels = transitions['transition'].tolist()
    x_pos = {t: i for i, t in enumerate(x_labels)}

    def series_for(data):
        """Map a party's shifts onto the shared axis; NaN where it has no value."""
        y = np.full(len(x_labels), np.nan)
        for t, s in zip(data['transition'], data['shift']):
            y[x_pos[t]] = s
        return y

    dem_y = series_for(dem_data)
    rep_y = series_for(rep_data)

    # Shared, data-driven y-limit with ~12% headroom so peak markers (which have
    # radius) never run off the top frame. Both panels share it so the magnitude
    # comparison between parties is honest (was a hard-coded 0.4 vs 0.6 mismatch
    # that clipped the data).
    data_max = float(np.nanmax(np.concatenate([dem_y, rep_y])))
    y_top = data_max * 1.12

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    panels = [
        (axes[0], dem_y, 'Democrat', 'o'),
        (axes[1], rep_y, 'Republican', 's'),
    ]
    for ax, y, party, marker in panels:
        color = PARTY_COLORS[party]
        ax.plot(range(len(x_labels)), y, marker=marker, linewidth=3,
                markersize=12, color=color, alpha=0.9)
        ax.fill_between(range(len(x_labels)), 0, y, alpha=0.15, color=color)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Period Transition', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_title(f'{party}: "{term.replace("_", " ")}"', fontsize=14, fontweight='bold',
                     color=color, pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=y_top)
        ax.margins(x=0.04)  # small x-pad so first/last markers aren't clipped

    axes[0].set_ylabel('Semantic Shift (Cosine Distance)', fontsize=11, fontweight='bold')

    # plt.suptitle(f'Partisan Shift Comparison: "{term}"', fontsize=16, fontweight='bold', y=0.96)

    # Adjust subplot spacing to prevent x-axis label cutoff
    plt.subplots_adjust(bottom=0.25, top=0.90, wspace=0.15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()

# Create side-by-side comparisons for key terms
comparison_terms = [
    'integrity',
    'fraud',
    'security',
    'election',
    'voter',
    'ballot',
    'mail',
    'access',
    'stolen',
    'trust'
]
for term in comparison_terms:
    print(f"\nPlotting comparative shift timeline: {term}")
    save_path = os.path.join(FIG_DIR, f'shift_comparison_{term}.png')
    plot_comparative_shift_timeline(all_shifts, term, save_path)

"""#Neighbor Overlap Analysis"""

def calculate_neighbor_overlap(dem_neighbors, rep_neighbors, k=10):
    """Calculate Jaccard similarity between neighbor sets"""
    dem_words = set([w for w, s in dem_neighbors[:k]])
    rep_words = set([w for w, s in rep_neighbors[:k]])

    if len(dem_words) == 0 or len(rep_words) == 0:
        return None

    intersection = dem_words.intersection(rep_words)
    union = dem_words.union(rep_words)

    jaccard = len(intersection) / len(union)
    overlap_count = len(intersection)

    return {
        'jaccard': jaccard,
        'overlap_count': overlap_count,
        'shared_terms': list(intersection)
    }

# Calculate overlap for all term-period combinations
overlap_data = []

for term in TARGET_TERMS:
    for period in periods:
        # Get neighbors
        dem_neighbors = partisan_neighbors['Democrat'].get(term, {}).get(period, [])
        rep_neighbors = partisan_neighbors['Republican'].get(term, {}).get(period, [])

        if not dem_neighbors or not rep_neighbors:
            continue

        overlap = calculate_neighbor_overlap(dem_neighbors, rep_neighbors)

        if overlap:
            overlap_data.append({
                'term': term,
                'period': period,
                'jaccard': overlap['jaccard'],
                'overlap_count': overlap['overlap_count'],
                'shared_terms': ', '.join(overlap['shared_terms'])
            })

overlap_df = pd.DataFrame(overlap_data)

# Save
overlap_path = os.path.join(OUTPUT_DIR, 'neighbor_overlap.csv')
overlap_df.to_csv(overlap_path, index=False)
print(f"✓ Overlap analysis saved to: {overlap_path}")

# Also save with the column names expected by the network visualization (step 4)
overlap_analysis_path = os.path.join(OUTPUT_DIR, 'neighbors_overlap_analysis.csv')
overlap_df.rename(columns={
    'period': 'year',
    'jaccard': 'jaccard_similarity',
    'overlap_count': 'shared_neighbors_count',
}).to_csv(overlap_analysis_path, index=False)
print(f"✓ Overlap analysis (step 4 schema) saved to: {overlap_analysis_path}")

# Summary statistics
print("\nTerms with LOWEST average overlap (most polarized):")
low_overlap = overlap_df.groupby('term')['jaccard'].mean().sort_values().head(10)
print(low_overlap)

print("\nTerms with HIGHEST average overlap (least polarized):")
high_overlap = overlap_df.groupby('term')['jaccard'].mean().sort_values(ascending=False).head(10)
print(high_overlap)

print("\nOverlap over time (averaging across terms):")
temporal_overlap = overlap_df.groupby('period')['jaccard'].mean().sort_index()
print(temporal_overlap)

# Visualize overlap over time
fig, ax = plt.subplots(figsize=(12, 6))

# Plot overall trend
ax.plot(temporal_overlap.index, temporal_overlap.values,
        'o-', linewidth=3, markersize=10, color='purple', label='Average Overlap')

# Plot selected terms
selected_terms = ['integrity', 'fraud', 'voter', 'election', 'ballot', 'mail', 'stolen', 'trust', 'access']
for term in selected_terms:
    term_overlap = overlap_df[overlap_df['term'] == term]
    if len(term_overlap) > 0:
        ax.plot(term_overlap['period'], term_overlap['jaccard'],
               'o--', linewidth=2, markersize=7, alpha=0.6, label=term)

ax.set_xlabel(period_label, fontsize=12, fontweight='bold')
ax.set_ylabel('Jaccard Similarity (Neighbor Overlap)', fontsize=12, fontweight='bold')
ax.set_title('Partisan Convergence/Divergence Over Time', fontsize=16, fontweight='bold')
ax.set_ylim([0, 1])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

overlap_plot_path = os.path.join(FIG_DIR, 'neighbor_overlap_timeline.png')
plt.tight_layout()
plt.savefig(overlap_plot_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {overlap_plot_path}")
plt.close()

"""#Neighbor Overlap - Zoomed Trend View"""
# Same Jaccard data as neighbor_overlap_timeline.png, replotted so convergence/
# divergence is actually legible. Three changes vs. the original:
#   1. y-axis autoscaled to the data (values live in ~0-0.2, not 0-1), so the
#      variation isn't crushed into the bottom 1/5 of the frame.
#   2. a shaded band = interquartile spread of the per-term Jaccard each year,
#      so you see dispersion without a tangle of dashed lines.
#   3. a linear trend line on the average, so the DIRECTION (converging vs.
#      diverging) is unambiguous and labeled with the slope.
fig, ax = plt.subplots(figsize=(12, 6))

# Spread band: 25th-75th percentile of per-term Jaccard within each year.
spread = overlap_df.groupby('period')['jaccard'].agg(
    lo=lambda s: np.percentile(s, 25),
    hi=lambda s: np.percentile(s, 75),
).sort_index()
ax.fill_between(spread.index, spread['lo'], spread['hi'],
                color='purple', alpha=0.15, label='Term spread (IQR)')

# Average overlap line (bold).
ax.plot(temporal_overlap.index, temporal_overlap.values,
        'o-', linewidth=3, markersize=9, color='purple', label='Average Overlap')

# Linear trend on the average -> direction of convergence/divergence.
years = np.array(temporal_overlap.index, dtype=float)
vals = temporal_overlap.values.astype(float)
slope, intercept = np.polyfit(years, vals, 1)
direction = 'diverging' if slope < 0 else 'converging'
ax.plot(years, slope * years + intercept, '--', color='black', linewidth=1.5,
        alpha=0.7, label=f'Trend: {slope:+.4f}/yr ({direction})')

# Autoscale y to the data with a little headroom (the key readability fix).
y_hi = float(np.nanmax(spread['hi'].values)) * 1.15
ax.set_ylim([0, y_hi])

ax.set_xlabel(period_label, fontsize=12, fontweight='bold')
ax.set_ylabel('Jaccard Similarity (Neighbor Overlap)', fontsize=12, fontweight='bold')
ax.set_title('Partisan Convergence/Divergence Over Time (zoomed)', fontsize=16, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

zoom_plot_path = os.path.join(FIG_DIR, 'neighbor_overlap_timeline_zoom.png')
plt.tight_layout()
plt.savefig(zoom_plot_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {zoom_plot_path}")
plt.close()

"""#Neighbor Overlap - Heatmap (term x year)"""
# The same Jaccard values as a term x year grid, so there are no overlapping
# lines at all: color encodes overlap. Darker/brighter cells = more shared
# neighbors (convergence); dark/low cells = divergence. Rows are sorted by mean
# overlap so the most-shared terms sit at the top. Blank cells = the term wasn't
# in both parties' vocab that year.
heat = overlap_df.pivot_table(index='term', columns='period', values='jaccard')
heat = heat.loc[heat.mean(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(14, max(6, len(heat) * 0.4)))
sns.heatmap(
    heat,
    cmap='magma',
    vmin=0,
    cbar_kws={'label': 'Jaccard Similarity (Neighbor Overlap)'},
    linewidths=0.5,
    linecolor='white',
    ax=ax,
)
ax.set_xlabel(period_label, fontsize=12, fontweight='bold')
ax.set_ylabel('Term', fontsize=12, fontweight='bold')
ax.set_title('Partisan Neighbor Overlap by Term and Year', fontsize=16, fontweight='bold')

heatmap_plot_path = os.path.join(FIG_DIR, 'neighbor_overlap_timeline_heatmap.png')
plt.tight_layout()
plt.savefig(heatmap_plot_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {heatmap_plot_path}")
plt.close()

"""#Generate Comprehensive Report"""

report_path = os.path.join(OUTPUT_DIR, 'partisan_diachronic_report.txt')

with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("PARTISAN DIACHRONIC WORD EMBEDDINGS ANALYSIS\n")
    f.write("Election Integrity Rhetoric: Democrats vs. Republicans\n")
    f.write("="*80 + "\n\n")

    f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")

    # Configuration
    f.write("CONFIGURATION\n")
    f.write("-"*80 + "\n")
    f.write(f"Time Binning: {TIME_BINNING}\n")
    f.write(f"Periods Analyzed: {len(periods)}\n")
    f.write(f"Period Range: {min(periods)} - {max(periods)}\n")
    f.write(f"Target Terms: {len(TARGET_TERMS)}\n")
    f.write(f"Word2Vec Dimensions: {W2V_PARAMS['vector_size']}\n")
    f.write(f"Word2Vec Seed: {SEED}\n\n")

    # Corpus statistics
    f.write("CORPUS STATISTICS\n")
    f.write("-"*80 + "\n")
    for _, row in stats_df.iterrows():
        f.write(f"\n{row['period']}:\n")
        f.write(f"  Democrat docs: {row['dem_docs']:,}\n")
        f.write(f"  Republican docs: {row['rep_docs']:,}\n")
        f.write(f"  Democrat vocab: {row['dem_vocab']:,}\n")
        f.write(f"  Republican vocab: {row['rep_vocab']:,}\n")
        f.write(f"  Shared vocab (within period): {row['shared_vocab']:,}\n")

    f.write(f"\nDemocrat shared vocab (across time): {len(dem_shared_vocab_list):,}\n")
    f.write(f"Republican shared vocab (across time): {len(rep_shared_vocab_list):,}\n\n")

    # Top within-party shifts
    f.write("\nLARGEST WITHIN-PARTY SEMANTIC SHIFTS\n")
    f.write("-"*80 + "\n")
    f.write("\nDemocrats:\n")
    top_dem = all_shifts[all_shifts['party'] == 'Democrat'].nlargest(15, 'shift')
    for _, row in top_dem.iterrows():
        f.write(f"  {row['term']:15s} {row['period1']} → {row['period2']}: {row['shift']:.3f}\n")

    f.write("\nRepublicans:\n")
    top_rep = all_shifts[all_shifts['party'] == 'Republican'].nlargest(15, 'shift')
    for _, row in top_rep.iterrows():
        f.write(f"  {row['term']:15s} {row['period1']} → {row['period2']}: {row['shift']:.3f}\n")

    # Asymmetric polarization (mean within-party shift; computed earlier)
    f.write("\n\nASYMMETRIC POLARIZATION (mean within-party shift)\n")
    f.write("-"*80 + "\n")
    f.write(f"  Democrat mean shift:   {dem_mean:.3f}  (n={len(dem_sh)})\n")
    f.write(f"  Republican mean shift: {rep_mean:.3f}  (n={len(rep_sh)})\n")
    f.write(f"  {larger} shifted {pct_diff:.0f}% more (95% CI: {ci_lo:.0f}%-{ci_hi:.0f}%)\n")
    f.write("  (pools across terms and transitions; if the CI spans a small value,\n")
    f.write("   report as indistinguishable rather than a headline asymmetry)\n")

    # Neighbor overlap
    f.write("\n\nNEIGHBOR OVERLAP ANALYSIS\n")
    f.write("-"*80 + "\n")
    f.write("\nTerms with LOWEST overlap (most polarized):\n")
    low_overlap = overlap_df.groupby('term')['jaccard'].mean().sort_values()
    for term, jaccard in low_overlap.head(10).items():
        f.write(f"  {term:15s}: {jaccard:.3f}\n")

    f.write("\nTerms with HIGHEST overlap (least polarized):\n")
    high_overlap = overlap_df.groupby('term')['jaccard'].mean().sort_values(ascending=False)
    for term, jaccard in high_overlap.head(10).items():
        f.write(f"  {term:15s}: {jaccard:.3f}\n")

    # Temporal trends
    f.write("\n\nTEMPORAL TRENDS\n")
    f.write("-"*80 + "\n")

    # Is overlap increasing or decreasing?
    temporal_overlap = overlap_df.groupby('period')['jaccard'].mean().sort_index()
    if len(temporal_overlap) >= 3:
        early_overlap = temporal_overlap.iloc[:len(temporal_overlap)//2].mean()
        late_overlap = temporal_overlap.iloc[len(temporal_overlap)//2:].mean()
        trend = "DECREASING" if late_overlap < early_overlap else "INCREASING"
        f.write(f"\nNeighbor overlap trend: {trend}\n")
        f.write(f"  Early periods average: {early_overlap:.3f}\n")
        f.write(f"  Late periods average: {late_overlap:.3f}\n")
        f.write(f"  Change: {late_overlap - early_overlap:+.3f}\n")

    # Output files
    f.write("\n\nOUTPUT FILES\n")
    f.write("-"*80 + "\n")
    f.write("Data files:\n")
    f.write("  - corpus_statistics.csv\n")
    f.write("  - partisan_semantic_shifts.csv\n")
    f.write("  - partisan_shift_asymmetry.csv\n")
    f.write("  - neighbor_overlap.csv\n")
    f.write("  - partisan_neighbors_over_time.txt\n")
    f.write("\nVisualizations:\n")
    f.write("  - democrat_shift_timeline.png (Democrat shifts over time)\n")
    f.write("  - republican_shift_timeline.png (Republican shifts over time)\n")
    f.write("  - shift_comparison_*.png (side-by-side shift comparisons)\n")
    f.write("  - neighbor_comparison_*.png (side-by-side neighbor heatmaps)\n")
    f.write("  - trajectories_*.png (2D semantic trajectories)\n")
    f.write("  - neighbor_overlap_timeline.png (convergence/divergence)\n")
    f.write("\nModels:\n")
    f.write("  - dem_model_*.bin (Democrat Word2Vec models)\n")
    f.write("  - rep_model_*.bin (Republican Word2Vec models)\n")


print("\n" + "="*80)
print("PARTISAN DIACHRONIC ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nKey findings to examine:")
print("1. partisan_neighbors_over_time.txt - How do Dems vs Reps use terms?")
print("3. neighbor_overlap.csv - Are parties converging or diverging?")
print("4. democrat_shift_timeline.png - Democrat semantic shifts over time")
print("5. republican_shift_timeline.png - Republican semantic shifts over time")
print("6. shift_comparison_*.png - Side-by-side shift comparisons")
print("7. neighbor_comparison_*.png - Visual comparison of neighbor evolution")
print("8. trajectories_*.png - Semantic paths diverging or parallel?")
print("\nNext steps:")
print("- Compare Democrat vs Republican shift magnitudes")
print("- Write up partisan findings for paper!")