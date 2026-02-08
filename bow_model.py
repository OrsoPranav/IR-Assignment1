"""
Part (b): Bag-of-Words (BoW) Model
====================================
Implements unigram and bigram BoW representations for a document corpus,
and supports query-document similarity comparison using cosine similarity.
"""

import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# N-gram extraction helpers
# ---------------------------------------------------------------------------
def get_unigrams(tokens: List[str]) -> List[str]:
    """Return unigram token list (identity)."""
    return tokens


def get_bigrams(tokens: List[str]) -> List[str]:
    """Return bigram token list as joined strings."""
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


# ---------------------------------------------------------------------------
# BoW representation
# ---------------------------------------------------------------------------
class BagOfWords:
    """
    Build a BoW model for a document corpus.

    Parameters
    ----------
    processed_docs : list of list of str
        Pre-processed token lists for each document.
    mode : {'unigram', 'bigram', 'both'}
        Which n-gram representation to use.
    """

    def __init__(self, processed_docs: List[List[str]], mode: str = "unigram"):
        self.mode = mode
        self.doc_ngrams: List[List[str]] = []
        self.vocab: Dict[str, int] = {}    # term -> index
        self.idf: Dict[str, float] = {}    # term -> idf
        self.tf_matrix: List[Dict[str, int]] = []   # per-doc term freqs
        self.num_docs = len(processed_docs)

        self._build(processed_docs)

    # ---- internal helpers --------------------------------------------------
    def _extract(self, tokens: List[str]) -> List[str]:
        if self.mode == "unigram":
            return get_unigrams(tokens)
        elif self.mode == "bigram":
            return get_bigrams(tokens)
        else:  # both
            return get_unigrams(tokens) + get_bigrams(tokens)

    def _build(self, processed_docs: List[List[str]]):
        # 1. Extract n-grams for every document
        for tokens in processed_docs:
            ngrams = self._extract(tokens)
            self.doc_ngrams.append(ngrams)

        # 2. Build vocabulary
        all_terms = set()
        for ngrams in self.doc_ngrams:
            all_terms.update(ngrams)
        self.vocab = {term: idx for idx, term in enumerate(sorted(all_terms))}

        # 3. Term-frequency vectors
        self.tf_matrix = [Counter(ngrams) for ngrams in self.doc_ngrams]

        # 4. IDF
        df = Counter()
        for tf in self.tf_matrix:
            for term in tf:
                df[term] += 1
        self.idf = {
            term: math.log((self.num_docs + 1) / (count + 1)) + 1
            for term, count in df.items()
        }

    # ---- public API -------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_tfidf_vector(self, tf: Dict[str, int]) -> np.ndarray:
        """Convert a term-frequency dict to a dense TF-IDF vector."""
        vec = np.zeros(self.vocab_size)
        for term, freq in tf.items():
            if term in self.vocab:
                idx = self.vocab[term]
                vec[idx] = freq * self.idf.get(term, 1.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def query_similarity(
        self, query_tokens: List[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Return the top-k most similar documents to a query.

        Parameters
        ----------
        query_tokens : list of str
            Already-preprocessed tokens of the query.
        top_k : int

        Returns
        -------
        list of (doc_index, cosine_similarity)
        """
        query_ngrams = self._extract(query_tokens)
        query_tf = Counter(query_ngrams)
        q_vec = self.get_tfidf_vector(query_tf)

        scores = []
        for doc_idx, doc_tf in enumerate(self.tf_matrix):
            d_vec = self.get_tfidf_vector(doc_tf)
            sim = float(np.dot(q_vec, d_vec))
            scores.append((doc_idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_bow_vector_raw(self, doc_idx: int) -> Dict[str, int]:
        """Return raw BoW (term-frequency) dict for a document."""
        return dict(self.tf_matrix[doc_idx])

    def compare_representations(
        self, query_tokens: List[str], top_k: int = 5
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Run unigram AND bigram retrieval on the same corpus and query.
        Returns dict with keys 'unigram' and 'bigram'.
        """
        results = {}
        original_mode = self.mode

        for mode in ("unigram", "bigram"):
            self.mode = mode
            results[mode] = self.query_similarity(query_tokens, top_k)

        self.mode = original_mode
        return results

    def summary(self) -> dict:
        """Return summary statistics."""
        total_tokens = sum(len(ng) for ng in self.doc_ngrams)
        return {
            "mode": self.mode,
            "num_documents": self.num_docs,
            "vocab_size": self.vocab_size,
            "total_ngram_tokens": total_tokens,
            "avg_tokens_per_doc": round(total_tokens / max(self.num_docs, 1), 2),
        }
