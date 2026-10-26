import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.patheffects as path_effects
import pandas as pd
import os

# Which Step-3 time binning to read. Step 3 writes its tidy CSVs into a
# {TIME_BINNING} subdirectory (e.g. .../by_party/year/neighbors_partisan.csv),
# so this MUST match the binning Step 3 was run with ('year' or 'congress').
TIME_BINNING = 'year'  # Options: 'congress', 'year'  (2-yr bins: ~2x docs/cell)
# by_party writes into a per-seed subdirectory ({TIME_BINNING}/{SEED}), so this
# must match the SEED of the by_party run whose neighbors you want to plot.
BY_PARTY_SEED = 42
BY_YEAR_DIR = f'./filtered_outputs/diachronic/by_year/{TIME_BINNING}'
BY_PARTY_DIR = f'./filtered_outputs/diachronic/by_party/{TIME_BINNING}/{BY_PARTY_SEED}'

#read csv as pandas data frame
df_temporal = pd.read_csv(os.path.join(BY_YEAR_DIR, 'neighbors_temporal.csv'))
df_partisan = pd.read_csv(os.path.join(BY_PARTY_DIR, 'neighbors_partisan.csv'))
df_overlap = pd.read_csv(os.path.join(BY_PARTY_DIR, 'neighbors_overlap_analysis.csv'))

# Set paths
BASE_DIR = './filtered_outputs/'
OUTPUT_DIR = BASE_DIR + 'network/'
CSV_DIR = BASE_DIR

# Create output directory if it doesn't exist
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARTY_COLORS = {
    'democrat': '#3498db',      # Blue
    'republican': '#e74c3c',    # Red
    'shared': '#9b59b6',        # Purple
    'dem_light': '#5dade2',     # Light blue
    'rep_light': '#ec7063'      # Light red
}

NETWORKS = [
    'access',
    'audit',
    'ballot',
    'certification',
    'election',
    'fraud',
    'illegal',
    'integrity',
    'mail',
    'rigged',
    'security',
    'stolen',
    'transparency',
    'trust',
    'voter',
]

def create_partisan_network(
    term,
    year,
    top_n=10,
    score_encoding='edge_weight',  # Default to edge_weight only
    min_node_size=500,
    max_node_size=2500,
    min_edge_width=0.5,
    max_edge_width=4,
    show_scores=True,
    layout_radius=0.7,
    output_path=None,
    figsize=(18, 9)
):
    """
    Create side-by-side partisan network comparison using CSV data

    Parameters:
    -----------
    term : str
        Target term to visualize (e.g., 'integrity', 'fraud')
    year : int
        Year to analyze
    top_n : int
        Number of top neighbors to show
    score_encoding : str
        How to encode similarity scores:
        - 'node_size': Larger nodes = higher similarity
        - 'edge_weight': Thicker edges = higher similarity
        - 'both': Use both (recommended)
    min_node_size, max_node_size : float
        Range for node sizes when score_encoding includes 'node_size'
    min_edge_width, max_edge_width : float
        Range for edge widths when score_encoding includes 'edge_weight'
    show_scores : bool
        Display similarity scores as text labels
    layout_radius : float
        Distance of neighbors from center (0.5-1.0 recommended)
    output_path : str
        Where to save the figure
    figsize : tuple
        Figure size in inches
    """

    # Filter data for this term and year
    mask = (df_partisan['term'] == term) & (df_partisan['year'] == year)
    term_year_data = df_partisan[mask]

    if len(term_year_data) == 0:
        # Not an error: the term simply has no neighbor rows for this year
        # (it didn't clear min_count in step 3's vocabulary that period).
        print(f"  - skipped '{term}' {year}: no data")
        return None

    dem_data = term_year_data[term_year_data['party'] == 'Democrat'].nsmallest(top_n, 'rank')
    rep_data = term_year_data[term_year_data['party'] == 'Republican'].nsmallest(top_n, 'rank')

    if len(dem_data) == 0 and len(rep_data) == 0:
        print(f"No neighbor data for '{term}' in {year}")
        return None

    # Convert to list of dicts format for compatibility with drawing code
    dem_neighbors = [
        {'word': row['neighbor'], 'score': row['similarity_score'], 'rank': int(row['rank'])}
        for _, row in dem_data.iterrows()
    ]
    rep_neighbors = [
        {'word': row['neighbor'], 'score': row['similarity_score'], 'rank': int(row['rank'])}
        for _, row in rep_data.iterrows()
    ]

    # Calculate shared neighbors
    dem_words = set([n['word'] for n in dem_neighbors])
    rep_words = set([n['word'] for n in rep_neighbors])
    shared_words = dem_words.intersection(rep_words)

    # Get Jaccard similarity from overlap data (or calculate)
    overlap_row = df_overlap[
        (df_overlap['term'] == term) &
        (df_overlap['year'] == year)
    ]

    if len(overlap_row) > 0:
        jaccard = overlap_row.iloc[0]['jaccard_similarity']
        overlap_count = overlap_row.iloc[0]['shared_neighbors_count']
    else:
        jaccard = len(shared_words) / len(dem_words.union(rep_words)) if (dem_words or rep_words) else 0
        overlap_count = len(shared_words)

    # Create figure
    fig, (ax_dem, ax_rep) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    # Main title
    fig.suptitle(
        f'K-nearest neighbors: "{term.replace("_", " ").title()}" ({year})',
        fontsize=20, fontweight='bold', y=0.98
    )

    def draw_network(ax, neighbors, party, title):
        """Draw one network (Democrat or Republican)"""

        if not neighbors:

            ax.text(0, 0, 'No data', ha='center', va='center',
                   fontsize=16, color='gray')
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            ax.axis('off')
            return

        # Calculate positions (circular layout)
        positions = {}
        positions[term] = np.array([0, 0])

        n_neighbors = len(neighbors)
        angles = np.pi / 2 - 2 * np.pi * np.arange(n_neighbors) / n_neighbors

        for i, neighbor in enumerate(neighbors):
            word = neighbor['word']
            positions[word] = layout_radius * np.array([
                np.cos(angles[i]),
                np.sin(angles[i])
            ])

        # Normalize scores for scaling
        scores = [n['score'] for n in neighbors]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1

        def normalize_score(score):
            """Normalize score to 0-1 range"""
            return (score - min_score) / score_range if score_range > 0 else 0.5

        CENTER_R = 0.18           # radius of the central target node
        NODE_R = 0.135            # neighbour-node radius (edge-weight mode)
        BADGE_GAP = 0.08          # gap between a node's rim and its score badge

        def node_radius(norm_score):
            # In node-size mode the radius encodes the score; else it's constant.
            if score_encoding in ('node_size', 'both'):
                return 0.10 + 0.05 * norm_score
            return NODE_R

        def wrap_label(text, max_chars=12):
            # Greedily pack words into short lines so long multi-word labels
            # stay inside the node instead of spilling over edges/badges.
            words = text.split()
            lines, cur = [], ''
            for w in words:
                if cur and len(cur) + 1 + len(w) > max_chars:
                    lines.append(cur)
                    cur = w
                else:
                    cur = f'{cur} {w}'.strip()
            if cur:
                lines.append(cur)
            return '\n'.join(lines)

        # Draw edges: trimmed to start at the center node's rim and stop at the
        # neighbour's rim, so a line never runs under a label or a score badge.
        for neighbor in neighbors:
            word = neighbor['word']
            score = neighbor['score']
            norm_score = normalize_score(score)

            # Edge styling based on score_encoding
            if score_encoding in ['edge_weight', 'both']:
                edge_width = min_edge_width + (max_edge_width - min_edge_width) * norm_score
                edge_alpha = 0.3 + 0.5 * norm_score
            else:
                edge_width = 2
                edge_alpha = 0.4

            # Color edges based on shared status
            if word in shared_words:
                edge_color = PARTY_COLORS['shared']
            else:
                light_key = 'dem_light' if party == 'democrat' else 'rep_light'
                edge_color = PARTY_COLORS[light_key]

            p = positions[word]
            dist = np.hypot(p[0], p[1])
            u = p / dist if dist > 0 else p
            start = u * CENTER_R
            end = p - u * node_radius(norm_score)

            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=edge_color,
                alpha=edge_alpha,
                linewidth=edge_width,
                zorder=1,
                solid_capstyle='round'
            )

        # Draw center node (target term)
        ax.add_patch(plt.Circle(
            (0, 0), CENTER_R,
            facecolor=PARTY_COLORS[party],
            edgecolor='black',
            linewidth=3,
            zorder=10,
            alpha=0.95
        ))

        # Center node label
        ax.text(
            0, 0,
            term.replace('_', ' ').replace(' ', '\n'),
            ha='center', va='center',
            fontsize=13,
            fontweight='bold',
            color='white',
            zorder=11
        )

        # Draw neighbor nodes + labels + score badges
        for neighbor in neighbors:
            word = neighbor['word']
            score = neighbor['score']
            norm_score = normalize_score(score)
            is_shared = word in shared_words

            # Node styling based on shared status
            if is_shared:
                node_color = PARTY_COLORS['shared']
                node_edge = 'black'
                text_color = 'black'
                text_weight = 'bold'
            else:
                node_color = 'white'
                node_edge = PARTY_COLORS[party]
                text_color = 'black'
                text_weight = 'normal'

            p = positions[word]
            r = node_radius(norm_score)
            dist = np.hypot(p[0], p[1])
            u = p / dist if dist > 0 else p

            ax.add_patch(plt.Circle(
                p, r,
                facecolor=node_color,
                edgecolor=node_edge,
                linewidth=2.5,
                zorder=5,
                alpha=0.95
            ))

            # Node label (kept inside the node; wrapped + sized to fit).
            display_word = wrap_label(word.replace('_', ' '))
            n_lines = display_word.count('\n') + 1
            longest = max(len(line) for line in display_word.split('\n'))
            label_fs = 9 if (longest <= 11 and n_lines <= 2) else 8

            text_obj = ax.text(
                p[0], p[1],
                display_word,
                ha='center', va='center',
                fontsize=label_fs,
                color=text_color,
                fontweight=text_weight,
                zorder=6
            )

            # White outline so the label reads over the node fill/border.
            if not is_shared:
                text_obj.set_path_effects([
                    path_effects.Stroke(linewidth=2, foreground='white'),
                    path_effects.Normal()
                ])

            # Score badge: placed just beyond the node's outer rim along the
            # radial ray. Because the edge stops at the rim, the badge sits in
            # empty space with neither a line nor the label behind it.
            if show_scores:
                badge_pos = p + u * (r + BADGE_GAP)
                ax.text(
                    badge_pos[0], badge_pos[1],
                    f'{score:.2f}',
                    ha='center', va='center',
                    fontsize=8,
                    fontweight='bold',
                    color=PARTY_COLORS[party],
                    zorder=7,
                    bbox=dict(
                        boxstyle='round,pad=0.25',
                        facecolor='white',
                        edgecolor=PARTY_COLORS[party],
                        linewidth=1,
                        alpha=0.95
                    )
                )

        # Styling. A little extra margin so the outer score badges never clip.
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title with party color
        title_color = PARTY_COLORS[party]
        ax.set_title(
            title,
            fontsize=16,
            fontweight='bold',
            color=title_color,
            pad=15
        )

        # Add subtle background circle
        circle = plt.Circle(
            (0, 0),
            layout_radius + 0.32,  # enclose the outer ring of score badges
            color=PARTY_COLORS[party],
            alpha=0.05,
            zorder=0
        )
        ax.add_patch(circle)

    # Draw both networks
    draw_network(ax_dem, dem_neighbors, 'democrat', 'Democrat Discourse')
    draw_network(ax_rep, rep_neighbors, 'republican', 'Republican Discourse')

    # # Add legend
    # legend_elements = [
    #     mpatches.Patch(
    #         facecolor=PARTY_COLORS['shared'],
    #         edgecolor='black',
    #         linewidth=1.5,
    #         label=f'Shared terms (n={overlap_count})'
    #     ),
    #     mpatches.Patch(
    #         facecolor='white',
    #         edgecolor=PARTY_COLORS['democrat'],
    #         linewidth=1.5,
    #         label='Democrat-unique'
    #     ),
    #     mpatches.Patch(
    #         facecolor='white',
    #         edgecolor=PARTY_COLORS['republican'],
    #         linewidth=1.5,
    #         label='Republican-unique'
    #     )
    # ]

    # fig.legend(
    #     handles=legend_elements,
    #     loc='lower center',
    #     ncol=3,
    #     fontsize=12,
    #     frameon=True,
    #     bbox_to_anchor=(0.5, -0.02),
    #     fancybox=True,
    #     shadow=True
    # )

    # # Add statistics text
    # stats_text = f'Semantic Overlap: {overlap_count}/{top_n} shared neighbors | Jaccard Similarity: {jaccard:.3f}'
    # if jaccard < 0.2:
    #     stats_text += ' (Severe Divergence)'
    # elif jaccard < 0.5:
    #     stats_text += ' (Moderate Divergence)'

    # fig.text(
    #     0.5, 0.01,
    #     stats_text,
    #     ha='center',
    #     fontsize=11,
    #     style='italic',
    #     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8)
    # )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    ax_dem.set_anchor('E')
    ax_rep.set_anchor('W')

    # Save
    if output_path:
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', facecolor='white',
                    pil_kwargs={'quality': 95})

    return fig



def create_key_term_comparisons(year=2024, output_dir=OUTPUT_DIR):
    """Generate partisan networks for key election integrity terms"""
    os.makedirs(output_dir, exist_ok=True)

    # NB: the last entry was 'election integrity' (space, not underscore) — it
    # never matched a token and silently produced nothing. Now 'election'.
    key_terms = ['integrity', 'fraud', 'security', 'voter', 'ballot', 'election']

    print(f"\n{'='*60}")
    print(f"Generating Partisan Network Comparisons for {year}")
    print(f"{'='*60}\n")

    for term in key_terms:
        print(f"Creating network for '{term}'...")

        fig = create_partisan_network(
            term=term,
            year=year,
            top_n=10,
            score_encoding='edge_weight',
            show_scores=True,
            output_path=f'{output_dir}/partisan_{term}_{year}.jpg'
        )

        if fig:
            plt.close(fig)
            print(f"  ✓ Complete\n")
        else:
            print(f"  ✗ Failed\n")

def create_temporal_comparison(term, years, output_dir=OUTPUT_DIR):
    """Create partisan comparisons across multiple years for one term"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Temporal Partisan Analysis: '{term}'")
    print(f"{'='*60}\n")

    for year in years:
        print(f"Year {year}...")

        fig = create_partisan_network(
            term=term,
            year=year,
            top_n=10,
            score_encoding='edge_weight',
            output_path=f'{output_dir}/partisan_{term}_{year}_temporal.jpg'
        )

        if fig:
            plt.close(fig)
            print(f"  ✓ Complete\n")

def compare_encoding_methods(term='integrity', year=2024, output_dir=OUTPUT_DIR):
    """Generate same network with different encoding methods for comparison"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Comparing Score Encoding Methods")
    print(f"{'='*60}\n")

    methods = [
        ('node_size', 'Node Size Only'),
        ('edge_weight', 'Edge Weight Only'),
        ('both', 'Both (Recommended)')
    ]

    for method, label in methods:
        print(f"Creating network with {label}...")

        fig = create_partisan_network(
            term=term,
            year=year,
            top_n=10,
            score_encoding=method,
            output_path=f'{output_dir}/encoding_{method}_{term}_{year}.jpg'
        )

        if fig:
            plt.close(fig)
            print(f"  ✓ Complete\n")

def generate_all_networks(output_dir=OUTPUT_DIR, terms=NETWORKS):
    """
    Generate a partisan network for every (term, year) combination that has data
    in df_partisan, restricted to the terms in `terms` (defaults to the NETWORKS
    list defined at the top of the file). It never asks for a term/year the step 3
    output doesn't contain, so every call that runs produces a figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Term-year pairs present in the data for the requested terms, ordered for
    # readable output. Filtering to `terms` keeps this to the NETWORKS of interest
    # rather than every term-year in the CSV.
    combos = (
        df_partisan[df_partisan['term'].isin(terms)][['term', 'year']]
        .drop_duplicates()
        .sort_values(['term', 'year'])
    )

    # Warn about any requested terms that have no rows at all (e.g. typo, or the
    # term never cleared min_count in step 3).
    missing = [t for t in terms if t not in set(df_partisan['term'].unique())]
    if missing:
        print(f"  note: no data for {missing} (skipped entirely)")

    print(f"\n{'='*60}")
    print(f"Generating partisan networks for {len(terms)} NETWORKS "
          f"({len(combos)} term-year combos)")
    print(f"{'='*60}\n")

    made, skipped = 0, 0
    for i, (_, row) in enumerate(combos.iterrows(), start=1):
        term = row['term']
        year = int(row['year'])  # year is float64 in the CSV; int -> clean filenames

        fig = create_partisan_network(
            term=term,
            year=year,
            top_n=10,
            score_encoding='edge_weight',
            show_scores=True,
            output_path=f'{output_dir}/partisan_{term}_{year}.jpg'
        )

        if fig:
            plt.close(fig)
            made += 1
            # ~13 s per figure at 1200 dpi; without this the run is silent
            # between the header and "Done" and looks hung.
            print(f"  [{i}/{len(combos)}] partisan_{term}_{year}.jpg", flush=True)
        else:
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Done: {made} figures written to {output_dir}, {skipped} skipped")
    print(f"{'='*60}")


if __name__ == '__main__':
    generate_all_networks()