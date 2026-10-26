import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm
import os

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set random seed
np.random.seed(42)

# Configure paths — override via env vars for VACC/local setups.
DATA_PATH = os.environ.get(
    'DATA_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combineddata.csv'),
)
OUTPUT_DIR = os.environ.get(
    'OUTPUT_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'filtered_outputs'),
)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seed terms from subject matter experts. Must match step_2b_filtering.ipynb,
# which does the actual tier filtering; here they only seed the Word2Vec
# lexicon expansion below.
CORE_SEED_TERMS = [
    "election integrity",
    "electoral integrity",
    "election fraud",
    "electoral fraud",
    "election crimes",
    "election integrity unit",
    "restoring faith in our elections",
    "securing the vote",
    "voter fraud",
    "voter integrity",
    "vote integrity",
    "vote accuracy",
    "illegal voting",
    "illegal vote",
    "non-citizen voting",
    "non-citizens vote",
    "photo id law",
    "voter identification",
    "voter id",
    "voter photo id",
    "photo id requirement",
    "election security",
    "for the people act",
    "voter suppress",
    "voting rights advancement act",
    "restoring faith in democracy",
    "fair elections",
    "voter suppression",
    "voter discrimination",
]

CONTEXT_SEED_TERMS = [
    "collusion",
    "certification",
    "certify",
    "shelby decision",
    "redistricting",
    "gerrymandering",
    "touchscreens",
    "stop the steal",
    "russian interference",
    "rigged",
    "russia investigation",
    "russian meddling",
    "foreign interference",
    "voting rights act",
    "freedom to vote act",
    "john lewis voting rights",
    "voter empowerment act",
    "electoral count",
    "election interference",
    "election subversion",
    "big lie",
    "2010 election",
    "save act",
]

SECONDARY_TERMS = [
    "mail ballot",
    "absentee ballot",
    "early ballot",
    "vote by mail",
    "mail-in ballot",
    "mail-in voting",
    "hand count",
    "es&s",
    "dominion voting",
    "vote centers",
    "voter registration",
    "ballot security",
    "ballot harvesting",
    "ballot drop box",
    "voting system",
    "voter roll",
    "election worker",
]

W2V_SIMILARITY_THRESHOLD = 0.6

print(f"Output directory: {OUTPUT_DIR}")

# Load data
if DATA_PATH.endswith('.csv'):
    df = pd.read_csv(DATA_PATH)

print(f"Initial shape: {df.shape}")

# Clean data and drop nans
df = df.dropna(axis=1, how='all')
df = df.dropna(axis=0, how='all')

#drop all columns that start with Unnamed:
df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]

#create new columns that converts the column 'Unix Timestamps' like 1262329230000 to date time
df['Date'] = pd.to_datetime(df['Unix Timestamp'], unit='ms', errors='coerce')

# Combine Body and Subject
df['Body_Subject'] = (
    df['Body'].fillna('').astype(str) + ' ' +
    df['Subject'].fillna('').astype(str)
).str.strip()

_BOILERPLATE = [
    r'click here(?: to[^.\n]{0,60})?',
    r'(?:open|view) (?:this )?(?:e-?mail|message|newsletter)(?: in[^.\n]{0,40})?',
    r'in its own browser(?: window)?',
    r'(?:open|view) (?:a )?plain[- ]text(?: version)?(?: of[^.\n]{0,40})?',
    r'(?:view|watch) accessible video', r'(?:information )?view accessible(?: version)?',
    r'(?:sign up for )?e-?news(?:letter)?(?: sign ?up)?',
    r'visit (?:my|our) website', r'forward to a friend',
    r'share on (?:facebook|twitter|x)\b', r'follow me on \w+',
    r'update (?:my )?(?:profile|subscription(?: options)?)',
    r'manage your subscription', r'unsubscribe(?: here)?', r'privacy policy',
    r'constant contact[^.\n]{0,40}', r'constituent services',
    r'issues? & legislation', r'how can we help', r'stay connected',
    r'\bdear (?:friend|constituent|neighbor)s?\b',
    r'\b(?:sincerely|warm(?:est)? regards|best regards|respectfully)\b[, ]*',
    # office-address footers, e.g. "1410 Longworth House Office Building, Washington, DC 20515"
    r'\d{3,4}\s+(?:longworth|cannon|rayburn)[^,\n]{0,40},?\s*washington,?\s*d\.?c\.?\s*20515(?:\s*us)?',
    r'\bwashington,?\s*d\.?c\.?\s*20515(?:\s*us)?',
    # phone numbers / urls / emails
    r'\b1[-.\s]?8\d\d[-.\s]?\d{3}[-.\s]?\d{4}\b', r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
    r'https?://\S+', r'www\.\S+', r'\S+@\S+\.\S+',
]
_BRE = re.compile('|'.join(_BOILERPLATE), flags=re.IGNORECASE)

def clean_boilerplate(text):
    """Strip templated newsletter chrome; collapse whitespace and non-ASCII."""
    if pd.isna(text):
        return text
    t = _BRE.sub(' ', str(text))
    t = re.sub(r'[^\x00-\x7F]+', ' ', t)   # mojibake / non-ASCII (stray box chars)
    t = re.sub(r'\s*\|\s*', ' ', t)        # leftover nav-menu pipes
    return re.sub(r'\s+', ' ', t).strip()

_orig_mean = df['Body_Subject'].str.len().mean()
df['Body_Subject'] = df['Body_Subject'].apply(clean_boilerplate)
print(f"Boilerplate cleanup: mean chars/doc "
      f"{_orig_mean:.0f} -> {df['Body_Subject'].str.len().mean():.0f}")

#drop columns Body and Subject
df = df.drop(columns=['Body', 'Subject'])

# Add metadata
df['text_length'] = df['Body_Subject'].str.len()

print(f"Final shape: {df.shape}")
print(f"Average text length: {df['text_length'].mean():.0f} characters")

"""#Lexicon Expansion with Word2Vec"""

# Tokenize corpus
tokenized_corpus = []
for text in tqdm(df['Body_Subject'].fillna(''), desc="Tokenizing"):
    text_lower = text.lower()
    tokens = word_tokenize(text_lower)
    tokenized_corpus.append(tokens)

# Train Word2Vec
w2v_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    epochs=5
)

print(f"Vocabulary size: {len(w2v_model.wv):,}")

# Expand seed terms
all_seeds = CORE_SEED_TERMS + CONTEXT_SEED_TERMS + SECONDARY_TERMS
expanded_lexicon = {}
missing_seeds = []

for seed in tqdm(all_seeds, desc="Expanding"):
    seed_processed = seed.lower().replace(' ', '_')

    if seed_processed in w2v_model.wv:
        similar = w2v_model.wv.most_similar(seed_processed, topn=20)
        expanded_lexicon[seed] = [
            (term.replace('_', ' '), score)
            for term, score in similar
            if score > W2V_SIMILARITY_THRESHOLD
        ]
    else:
        missing_seeds.append(seed)

# Display sample expansions
for seed in list(expanded_lexicon.keys())[:5]:
    print(f"\n{seed}:")
    for term, score in expanded_lexicon[seed][:5]:
        print(f"  {term} ({score:.3f})")

if missing_seeds:
    print(f"\n Warning: {len(missing_seeds)} seeds not found in vocabulary")

# Save full expansion list
lexicon_path = os.path.join(OUTPUT_DIR, 'expanded_lexicon_FOR_REVIEW.txt')
with open(lexicon_path, 'w') as f:
    f.write("EXPANDED LEXICON - MANUAL REVIEW\n")
    f.write("=" * 80 + "\n\n")
    for seed, similar_terms in sorted(expanded_lexicon.items()):
        f.write(f"\n{seed}:\n")
        for term, score in similar_terms:
            f.write(f"  {term} ({score:.3f})\n")

"""#Mine Phrases"""

# Build bigram and trigram models
bigram = Phrases(
    tokenized_corpus,
    min_count=5,
    threshold=0.5,
    scoring='npmi'
)

trigram = Phrases(
    bigram[tokenized_corpus],
    min_count=5,
    threshold=0.5,
    scoring='npmi'
)

# Extract phrases containing election-related keywords
seed_keywords_lower = ['election', 'voter', 'vote', 'ballot', 'fraud', 'integrity', 'security']

all_phrases = set()

# Sample corpus for efficiency
sample_size = min(10000, len(tokenized_corpus))
sample_corpus = tokenized_corpus[:sample_size]

# Apply bigram model and extract phrases
for sentence_transformed in bigram[sample_corpus]:
    for token in sentence_transformed:
        if '_' in token: # This indicates a phrase formed by the Phrases model
            phrase_clean = token.replace('_', ' ')
            if any(kw in phrase_clean.lower() for kw in seed_keywords_lower):
                all_phrases.add(phrase_clean)

# Apply trigram model (which is built on bigrams) and extract phrases
for sentence_transformed in trigram[bigram[sample_corpus]]:
    for token in sentence_transformed:
        if '_' in token:
            phrase_clean = token.replace('_', ' ')
            if any(kw in phrase_clean.lower() for kw in seed_keywords_lower):
                all_phrases.add(phrase_clean)

relevant_phrases = sorted(list(all_phrases)) # Convert set to list for sorting

print(f"\nDiscovered {len(relevant_phrases)} relevant phrases")
print("\nSample phrases:")
for phrase in relevant_phrases[:20]:
    print(f"  - {phrase}")

# Save phrases
phrases_path = os.path.join(OUTPUT_DIR, 'mined_phrases.txt')
with open(phrases_path, 'w') as f:
    for phrase in relevant_phrases:
        f.write(f"{phrase}\n")

"""#Semantic Encoding"""

# Load sentence transformer
print("\nLoading sentence transformer model...")
encoder = SentenceTransformer('all-mpnet-base-v2')

# Define semantic queries
semantic_queries = [
    "concerns about election integrity and voter fraud",
    "securing elections and preventing illegal voting and ballot harvesting",
    "election security and ballot verification",
    "voter identification and registration fraud",
    "noncitizen voting and proof-of-citizenship requirements to register and vote",
    "rigged elections and stolen votes",
    "election administration and certification processes",
    "mail-in ballot security, absentee voting, and ballot postmark deadlines",
    "voting machines and election technology security",
    "election denialism and false fraud claims",
    "partisan election rhetoric and delegitimization",
    "voter suppression and election access",
    "ballot harvesting and election crimes",
    "redrawing district lines to gain partisan political advantage",
    "redistricting and gerrymandering to dilute minority voting power",
    "foreign interference and Russian meddling in U.S. elections",
    "protecting elections from foreign influence and disinformation",
    "government units with police powers investigating election crimes",
    "claims that elections were stolen through widespread coordinated fraud",
]

print(f"Encoding {len(semantic_queries)} query templates...")
query_embeddings = encoder.encode(semantic_queries, show_progress_bar=False)

# Encode all newsletters
print(f"\nEncoding {len(df)} newsletters...")
text_embeddings = encoder.encode(
    df['Body_Subject'].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Calculate semantic similarities
similarities = util.cos_sim(text_embeddings, query_embeddings).numpy()
semantic_scores = similarities.max(axis=1)

# Add to dataframe
df['semantic_score'] = semantic_scores

df.to_csv(os.path.join(OUTPUT_DIR, 'semantic_scores_added.csv'), index=False)

# Persist the encoder artifacts alongside the scored CSV so
# step_2b_filtering.ipynb doesn't have to re-encode.
w2v_model.save(os.path.join(OUTPUT_DIR, 'word2vec_model.bin'))
np.save(os.path.join(OUTPUT_DIR, 'text_embeddings.npy'), text_embeddings)
np.save(os.path.join(OUTPUT_DIR, 'query_embeddings.npy'), query_embeddings)
np.save(os.path.join(OUTPUT_DIR, 'semantic_scores.npy'), semantic_scores)

print(f"\nEncoding complete. Artifacts written to: {OUTPUT_DIR}")
print("Next step: run step_2b_filtering.ipynb to apply tier filtering.")