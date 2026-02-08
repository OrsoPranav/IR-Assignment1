"""
Part (a): Preprocessing Pipeline for IR
========================================
Implements a multi-step text preprocessing pipeline and reports
the effect of each technique on vocabulary size.

Steps:
  1. Raw tokenization (whitespace / regex)
  2. Lowercasing
  3. Punctuation & special-character removal
  4. Stop-word removal
  5. Stemming  (Porter)
  6. Lemmatization (WordNet)
"""

import re
import time
from collections import Counter
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Ensure NLTK resources are available
# ---------------------------------------------------------------------------
for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Individual preprocessing steps
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Basic word tokenization using NLTK."""
    return word_tokenize(text)


def to_lowercase(tokens: List[str]) -> List[str]:
    return [t.lower() for t in tokens]


def remove_punctuation(tokens: List[str]) -> List[str]:
    """Keep only tokens that contain at least one alphanumeric character."""
    return [t for t in tokens if re.search(r"[A-Za-z0-9]", t)]


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP_WORDS]


def stem_tokens(tokens: List[str]) -> List[str]:
    return [STEMMER.stem(t) for t in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    return [LEMMATIZER.lemmatize(t) for t in tokens]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    ("1. Raw Tokenization", tokenize),
    ("2. Lowercasing", to_lowercase),
    ("3. Punctuation Removal", remove_punctuation),
    ("4. Stop-word Removal", remove_stopwords),
    ("5. Stemming (Porter)", stem_tokens),
]

PIPELINE_STEPS_LEMMA = [
    ("1. Raw Tokenization", tokenize),
    ("2. Lowercasing", to_lowercase),
    ("3. Punctuation Removal", remove_punctuation),
    ("4. Stop-word Removal", remove_stopwords),
    ("5. Lemmatization (WordNet)", lemmatize_tokens),
]


def run_pipeline(
    documents: List[str],
    pipeline=None,
    verbose: bool = False,
) -> Tuple[Dict[str, dict], List[List[str]]]:
    """
    Run the preprocessing pipeline on a list of document strings.

    Returns
    -------
    stats : dict
        Mapping step_name -> {
            'total_tokens': int,
            'vocab_size': int,
            'time_seconds': float
        }
    processed_docs : list[list[str]]
        Final tokens for every document.
    """
    if pipeline is None:
        pipeline = PIPELINE_STEPS

    # Initialize: each document is still a raw string
    current = documents  # list[str] for step-0, list[list[str]] afterwards
    stats = {}

    for step_name, func in pipeline:
        t0 = time.perf_counter()
        if step_name.startswith("1."):
            # tokenize expects str -> list[str]
            current = [func(doc) for doc in current]
        else:
            current = [func(tokens) for tokens in current]
        elapsed = time.perf_counter() - t0

        all_tokens = [tok for doc in current for tok in doc]
        vocab = set(all_tokens)
        stats[step_name] = {
            "total_tokens": len(all_tokens),
            "vocab_size": len(vocab),
            "time_seconds": round(elapsed, 4),
        }
        if verbose:
            print(
                f"  {step_name:40s} | tokens={len(all_tokens):>8,} | "
                f"vocab={len(vocab):>6,} | time={elapsed:.4f}s"
            )

    return stats, current


def preprocess_text(text: str) -> List[str]:
    """Convenience: full preprocessing on a single string (stemming path)."""
    tokens = tokenize(text)
    tokens = to_lowercase(tokens)
    tokens = remove_punctuation(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return tokens
