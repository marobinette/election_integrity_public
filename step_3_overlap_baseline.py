# -*- coding: utf-8 -*-
"""step_3_overlap_baseline.py — noise floor for the partisan Jaccard signal.
Usage
-----
    PYTHONHASHSEED=0 python step_3_overlap_baseline.py
"""

import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ------------------------------------------------------------------ Configuration
SEED = 42                 # matches the by_party run this baseline calibrates
TIME_BINNING = 'year'     # keep in step with by_party / aligned_divergence
N_SPLITS = 20             # random partitions per period; 3 for a smoke test
RUN_PERMUTATION = True    # the `perm` arm; costs 2 of the 6 models per split

NEIGHBOR_K = 25           # neighbors extracted, matching by_party's get_neighbors
JACCARD_K = 10            # neighbors actually compared, matching its overlap call

# A period is analyzable only if each group can hold this many documents. Below
# ~40 the models are too degenerate for the floor to mean anything.
MIN_DOCS_PER_GROUP = 40

# Must match TARGET_TERMS in by_party and aligned_divergence: it is also the
# phrasing protect set, so a different list means a different token stream.
TARGET_TERMS = [
    'fraud', 'integrity', 'security', 'voter',
    'ballot', 'election', 'certification', 'audit',
    'stolen', 'rigged', 'access', 'mail',
    'illegal', 'legitimate', 'trust', 'transparency',
]

# Same as by_party: 50-dim skip-gram, 40 epochs, workers=1 for reproducibility.
W2V_PARAMS = dict(vector_size=50, window=5, min_count=5, workers=1,
                  epochs=40, sg=1, negative=5, alpha=0.025, min_alpha=0.0001)

DATA_PATH = './filtered_outputs/all_filtered_combined.csv'
OUTPUT_DIR = f'./filtered_outputs/diachronic/overlap_baseline/{TIME_BINNING}'
PERIOD_LABEL = 'Year' if TIME_BINNING == 'year' else 'Congress'

# ------------------------------------------------------------------ Preprocess
stop_words = set(stopwords.words('english'))
stop_words.update({'would','could','should','may','might','must','said','says',
    'say','told','tell','get','got','getting','make','made','making','one','two',
    'three','first','second','last','also','however','therefore','thus','us','amp',
    'http','https','www'})

def preprocess_text(text):
    if pd.isna(text): return []
    return [t for t in word_tokenize(str(text).lower())
            if t.isalpha() and len(t) > 2 and t not in stop_words]

def year_to_congress(y):
    return None if pd.isna(y) or y < 2011 else 112 + (int(y) - 2011) // 2

def get_neighbors(model, word, k=25):
    """k nearest neighbors (no Procrustes needed; see docstring)."""
    if word not in model.wv:
        return []
    return model.wv.most_similar(word, topn=k)

def calculate_neighbor_overlap(neighbors_a, neighbors_b, k=10):
    """Top-k Jaccard, identical to by_party's calculate_neighbor_overlap."""
    words_a = set(w for w, s in neighbors_a[:k])
    words_b = set(w for w, s in neighbors_b[:k])
    if len(words_a) == 0 or len(words_b) == 0:
        return None
    intersection = words_a & words_b
    return {'jaccard': len(intersection) / len(words_a | words_b),
            'overlap_count': len(intersection),
            'shared_terms': list(intersection)}


# ------------------------------------------------------------------ Step 1: data
def load_corpus():
    """Load, filter and tokenize exactly as the by-party pipeline does.

    Every step of this must match step_3_diachronic_by_party.py, or the floor
    would be measured on a different corpus than the signal it calibrates: same
    party filter, same >=10-token document filter, same protected phrasing.
    """
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Congress'] = df['Year'].apply(year_to_congress)
    df['time_period'] = df['Congress'] if TIME_BINNING == 'congress' else df['Year']
    df = df[df['Party'].isin(['Democrat', 'Republican'])].copy()
    df = df.dropna(subset=['time_period'])
    df['time_period'] = df['time_period'].astype(int)

    print("Tokenizing + phrasing ...")
    df['tokens'] = df['Body_Subject'].apply(preprocess_text)
    df = df[df['tokens'].str.len() >= 10].copy()
    big = Phrases(df['tokens'].tolist(), min_count=5, threshold=10, delimiter='_')
    tri = Phrases([big[t] for t in df['tokens']], min_count=5, threshold=10, delimiter='_')

    # Undo any merge that swallows a target term (same rule as by_party).
    protect = set(TARGET_TERMS)
    def phrase(tokens):
        out = []
        for tok in tri[big[tokens]]:
            parts = tok.split('_')
            out.extend(parts if any(p in protect for p in parts) else [tok])
        return out
    df['tf'] = df['tokens'].apply(phrase)
    return df


# ------------------------------------------------- Step 2: the two primitives
def train(docs):
    """One model, trained the way every step-3 model is trained.

    SEED is held constant across all models here on purpose. The by-party
    pipeline trains its Democrat and Republican models with the same seed, so
    fixing it reproduces that condition: the floor then reflects sampling
    variation, which is the quantity the cross-party number also contains, and
    not extra initialization variation the real measurement never had.
    """
    return Word2Vec(docs, seed=SEED, **W2V_PARAMS)


def overlap(model_a, model_b, term):
    """Top-JACCARD_K neighbor Jaccard for `term` between two models.

    Returns None when the term missed min_count in either model — the same rule
    the by-party overlap uses, so a term absent there is absent here too. No
    alignment: neighbor ranking is rotation-invariant (see module docstring).
    """
    if term not in model_a.wv or term not in model_b.wv:
        return None
    na = get_neighbors(model_a, term, k=NEIGHBOR_K)
    nb = get_neighbors(model_b, term, k=NEIGHBOR_K)
    return calculate_neighbor_overlap(na, nb, k=JACCARD_K)


# ------------------------------------------------------------ Step 3: partitions
def draw_groups(dem_docs, rep_docs, n, rng):
    """Draw the disjoint, equal-sized document groups for one split.

    Returns a dict of six groups of exactly n documents each. demA/demB are
    disjoint within Democrats, repA/repB within Republicans, and permA/permB are
    a random split of the two parties pooled — which is what makes `perm` a null
    for the party label rather than for sampling.
    """
    d = rng.permutation(len(dem_docs))
    r = rng.permutation(len(rep_docs))
    groups = {
        'demA': [dem_docs[i] for i in d[:n]],
        'demB': [dem_docs[i] for i in d[n:2 * n]],
        'repA': [rep_docs[i] for i in r[:n]],
        'repB': [rep_docs[i] for i in r[n:2 * n]],
    }
    if RUN_PERMUTATION:
        pool = dem_docs + rep_docs
        p = rng.permutation(len(pool))
        groups['permA'] = [pool[i] for i in p[:n]]
        groups['permB'] = [pool[i] for i in p[n:2 * n]]
    return groups


# ------------------------------------------------------------- Step 4: main loop
def run_analysis():
    df = load_corpus()
    rows = []

    for period in sorted(df['time_period'].unique()):
        dem_docs = df[(df.time_period == period) & (df.Party == 'Democrat')]['tf'].tolist()
        rep_docs = df[(df.time_period == period) & (df.Party == 'Republican')]['tf'].tolist()

        # One common group size for all six groups, so corpus size cannot
        # masquerade as signal. The smaller party binds: with 391 Dem and 779
        # Rep in 2024, every group is 195 documents.
        n = min(len(dem_docs), len(rep_docs)) // 2
        if n < MIN_DOCS_PER_GROUP:
            print(f"  {period}: n={n} per group (Dem={len(dem_docs)} "
                  f"Rep={len(rep_docs)}) - EXCLUDED")
            continue

        print(f"  {period}: Dem={len(dem_docs)} Rep={len(rep_docs)} -> "
              f"n={n} per group x {N_SPLITS} splits")

        for b in range(N_SPLITS):
            # Seeded per (period, split) so any one cell reproduces on its own,
            # independent of iteration order or a partial re-run.
            rng = np.random.default_rng([SEED, int(period), b])
            g = draw_groups(dem_docs, rep_docs, n, rng)

            models = {name: train(docs) for name, docs in g.items()}

            pairs = [('within_dem', 'demA', 'demB'),
                     ('within_rep', 'repA', 'repB'),
                     ('cross_half', 'demA', 'repA')]
            if RUN_PERMUTATION:
                pairs.append(('perm', 'permA', 'permB'))

            for comparison, a, bb in pairs:
                for term in TARGET_TERMS:
                    o = overlap(models[a], models[bb], term)
                    if o is None:
                        continue          # term missed min_count in a group
                    rows.append({'term': term, 'period': int(period), 'split': b,
                                 'comparison': comparison, 'jaccard': o['jaccard'],
                                 'overlap_count': o['overlap_count'],
                                 'n_docs_per_group': n})

    return pd.DataFrame(rows)


# --------------------------------------------------------------- Step 5: report
def write_report(out, path):
    """Headline floor-vs-signal comparison, then the per-term breakdown.

    Two things a reader needs and the old report did not carry: an interval
    (we now have N_SPLITS draws per cell, so the floor has a spread, not just a
    mean) and per-term coverage (halving the data pushes sparse terms below
    min_count, so some terms rest on far fewer observations than others).
    """
    def stat(comparison, term=None):
        s = out[out['comparison'] == comparison]
        if term is not None:
            s = s[s['term'] == term]
        return s['jaccard']

    within = out[out['comparison'].isin(['within_dem', 'within_rep'])]['jaccard']
    cross = stat('cross_half')

    lines = ['=' * 78,
             'WITHIN-PARTY OVERLAP BASELINE (noise floor for the partisan Jaccard signal)',
             '=' * 78,
             f'Time binning: {TIME_BINNING}   Splits per period: {N_SPLITS}   '
             f'Seed: {SEED}   Size-matched: yes',
             f'Groups of n = min(|Dem|,|Rep|)//2 documents; top-{JACCARD_K} Jaccard.',
             '',
             'HEADLINE (mean [5th-95th pct] over all terms, periods and splits):']

    def fmt(label, s):
        if len(s) == 0:
            return f'  {label:<38}: (no observations)'
        return (f'  {label:<38}: {s.mean():.3f}  '
                f'[{s.quantile(0.05):.3f}-{s.quantile(0.95):.3f}]  n={len(s)}')

    lines += [fmt('within-party (same party, halves)', within),
              fmt('cross-party  (Dem half vs Rep half)', cross)]
    if RUN_PERMUTATION:
        lines.append(fmt('permuted    (party label shuffled)', stat('perm')))
    if len(cross) and cross.mean() > 0:
        lines.append(f'  {"ratio within/cross":<38}: {within.mean() / cross.mean():.2f}x')

    lines += ['',
              'How to read: if within >> cross at matched size, the partisan gap is real',
              'framing divergence; if within ~= cross, the low cross-party overlap is',
              'indistinguishable from small-corpus model noise. Report the partisan',
              'finding RELATIVELY (vs. the floor), never as an absolute "near-zero',
              'overlap". The `perm` row is the stricter test: it asks whether the party',
              'partition beats an arbitrary partition of the same documents.',
              '',
              'PER-TERM (mean across periods and splits, sorted by gap = within - cross):',
              f'  {"term":<16}{"within":>9}{"cross":>9}{"perm":>9}{"gap":>9}'
              f'{"n_cross":>9}{"periods":>9}']

    per_term = []
    for term in TARGET_TERMS:
        w = pd.concat([stat('within_dem', term), stat('within_rep', term)])
        c = stat('cross_half', term)
        p = stat('perm', term) if RUN_PERMUTATION else pd.Series(dtype=float)
        per_term.append({
            'term': term,
            'within': w.mean() if len(w) else np.nan,
            'cross': c.mean() if len(c) else np.nan,
            'perm': p.mean() if len(p) else np.nan,
            'gap': (w.mean() - c.mean()) if len(w) and len(c) else np.nan,
            'n_cross': len(c),
            'periods': out[(out['term'] == term) &
                           (out['comparison'] == 'cross_half')]['period'].nunique(),
        })
    pt = pd.DataFrame(per_term).sort_values('gap', ascending=False)

    for _, r in pt.iterrows():
        lines.append(f"  {r['term']:<16}{r['within']:>9.3f}{r['cross']:>9.3f}"
                     f"{r['perm']:>9.3f}{r['gap']:>9.3f}"
                     f"{int(r['n_cross']):>9}{int(r['periods']):>9}")

    lines += ['',
              'Terms with gap ~<= 0 have NO partisan signal above sampling noise; treat',
              'them qualitatively or drop them from quantitative claims. `periods` is how',
              'many years the term survived min_count in the halved corpora — a term with',
              'few periods rests on thin evidence even when its gap looks large.']

    report = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(report + '\n')
    pt.to_csv(path.replace('_report.txt', '_by_term.csv'), index=False)
    return report


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = run_analysis()

    csv_path = os.path.join(OUTPUT_DIR, 'overlap_baseline.csv')
    out.to_csv(csv_path, index=False)
    print(f"\nRaw observations -> {csv_path}  ({len(out)} rows)")

    print('\n' + write_report(out, os.path.join(OUTPUT_DIR,
                                                'overlap_baseline_report.txt')))
    print(f"\nOutputs -> {OUTPUT_DIR}/")
