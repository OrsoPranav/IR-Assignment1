"""
Part (c): Inverted Index with Three Data Structures
=====================================================
Implements the inverted index using:
  1. Python dict  (hash-map)
  2. Sorted list  (array-based, binary search)
  3. BST / balanced tree (via sortedcontainers.SortedDict)

Each variant exposes the same interface and is benchmarked for:
  - Storage size (memory)
  - Build time
  - Insertion
  - Retrieval (lookup)
  - Deletion
  - Update
"""

import bisect
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# ============================================================================
# Abstract base
# ============================================================================
class InvertedIndexBase(ABC):
    """Common interface for all inverted-index variants."""

    @abstractmethod
    def build(self, processed_docs: List[List[str]]) -> None:
        """Build the index from a list of tokenised documents."""

    @abstractmethod
    def search(self, term: str) -> List[int]:
        """Return posting list (sorted doc-ids) for *term*."""

    @abstractmethod
    def insert(self, term: str, doc_id: int) -> None:
        """Add *doc_id* to the posting list of *term*."""

    @abstractmethod
    def delete(self, term: str, doc_id: int) -> None:
        """Remove *doc_id* from the posting list of *term*."""

    @abstractmethod
    def update(self, old_term: str, new_term: str, doc_id: int) -> None:
        """Move *doc_id* from *old_term*'s list to *new_term*'s list."""

    @abstractmethod
    def vocab_size(self) -> int:
        """Number of distinct terms in the index."""

    @abstractmethod
    def memory_bytes(self) -> int:
        """Approximate memory consumption in bytes."""

    @abstractmethod
    def all_terms(self) -> List[str]:
        """Return all terms in the index."""


# ============================================================================
# 1. Hash-Map (Python dict) based index
# ============================================================================
class DictInvertedIndex(InvertedIndexBase):
    """Inverted index backed by a plain Python dict → set."""

    name = "HashMap (dict)"

    def __init__(self):
        self._index: Dict[str, set] = defaultdict(set)

    def build(self, processed_docs: List[List[str]]) -> None:
        self._index = defaultdict(set)
        for doc_id, tokens in enumerate(processed_docs):
            for token in tokens:
                self._index[token].add(doc_id)

    def search(self, term: str) -> List[int]:
        return sorted(self._index.get(term, set()))

    def insert(self, term: str, doc_id: int) -> None:
        self._index[term].add(doc_id)

    def delete(self, term: str, doc_id: int) -> None:
        if term in self._index:
            self._index[term].discard(doc_id)
            if not self._index[term]:
                del self._index[term]

    def update(self, old_term: str, new_term: str, doc_id: int) -> None:
        self.delete(old_term, doc_id)
        self.insert(new_term, doc_id)

    def vocab_size(self) -> int:
        return len(self._index)

    def memory_bytes(self) -> int:
        size = sys.getsizeof(self._index)
        for term, postings in self._index.items():
            size += sys.getsizeof(term) + sys.getsizeof(postings)
            for doc_id in postings:
                size += sys.getsizeof(doc_id)
        return size

    def all_terms(self) -> List[str]:
        return list(self._index.keys())

    def get_postings_set(self, term: str) -> set:
        return self._index.get(term, set())


# ============================================================================
# 2. Sorted-List (array + binary-search) based index
# ============================================================================
class SortedListInvertedIndex(InvertedIndexBase):
    """
    Inverted index where the vocabulary is stored in a sorted list and
    posting lists are sorted arrays.  Look-ups use binary search.
    """

    name = "SortedList (array + bisect)"

    def __init__(self):
        self._terms: List[str] = []                # sorted term list
        self._postings: Dict[str, List[int]] = {}  # term -> sorted doc_id list

    def build(self, processed_docs: List[List[str]]) -> None:
        tmp: Dict[str, set] = defaultdict(set)
        for doc_id, tokens in enumerate(processed_docs):
            for token in tokens:
                tmp[token].add(doc_id)
        self._terms = sorted(tmp.keys())
        self._postings = {t: sorted(tmp[t]) for t in self._terms}

    def _find_term(self, term: str) -> int:
        """Return index in _terms via bisect; -1 if absent."""
        idx = bisect.bisect_left(self._terms, term)
        if idx < len(self._terms) and self._terms[idx] == term:
            return idx
        return -1

    def search(self, term: str) -> List[int]:
        idx = self._find_term(term)
        if idx == -1:
            return []
        return list(self._postings[term])

    def insert(self, term: str, doc_id: int) -> None:
        idx = self._find_term(term)
        if idx == -1:
            bisect.insort(self._terms, term)
            self._postings[term] = [doc_id]
        else:
            plist = self._postings[term]
            pos = bisect.bisect_left(plist, doc_id)
            if pos >= len(plist) or plist[pos] != doc_id:
                bisect.insort(plist, doc_id)

    def delete(self, term: str, doc_id: int) -> None:
        idx = self._find_term(term)
        if idx == -1:
            return
        plist = self._postings[term]
        pos = bisect.bisect_left(plist, doc_id)
        if pos < len(plist) and plist[pos] == doc_id:
            plist.pop(pos)
        if not plist:
            self._terms.pop(idx)
            del self._postings[term]

    def update(self, old_term: str, new_term: str, doc_id: int) -> None:
        self.delete(old_term, doc_id)
        self.insert(new_term, doc_id)

    def vocab_size(self) -> int:
        return len(self._terms)

    def memory_bytes(self) -> int:
        size = sys.getsizeof(self._terms) + sys.getsizeof(self._postings)
        for t in self._terms:
            size += sys.getsizeof(t)
            plist = self._postings[t]
            size += sys.getsizeof(plist)
            for d in plist:
                size += sys.getsizeof(d)
        return size

    def all_terms(self) -> List[str]:
        return list(self._terms)

    def get_postings_set(self, term: str) -> set:
        return set(self.search(term))


# ============================================================================
# 3. BST / Skip-list (via a hand-written BST for pedagogical value)
# ============================================================================
class _BSTNode:
    """Node for a simple (unbalanced) binary search tree."""
    __slots__ = ("key", "postings", "left", "right")

    def __init__(self, key: str):
        self.key = key
        self.postings: set = set()
        self.left: Optional["_BSTNode"] = None
        self.right: Optional["_BSTNode"] = None


class BSTInvertedIndex(InvertedIndexBase):
    """Inverted index backed by a hand-written Binary Search Tree."""

    name = "BST (Binary Search Tree)"

    def __init__(self):
        self._root: Optional[_BSTNode] = None
        self._size: int = 0

    # ---- BST helpers -------------------------------------------------------
    def _insert_node(self, node: Optional[_BSTNode], key: str) -> _BSTNode:
        if node is None:
            self._size += 1
            return _BSTNode(key)
        if key < node.key:
            node.left = self._insert_node(node.left, key)
        elif key > node.key:
            node.right = self._insert_node(node.right, key)
        return node

    def _find_node(self, node: Optional[_BSTNode], key: str) -> Optional[_BSTNode]:
        if node is None:
            return None
        if key == node.key:
            return node
        elif key < node.key:
            return self._find_node(node.left, key)
        else:
            return self._find_node(node.right, key)

    def _min_node(self, node: _BSTNode) -> _BSTNode:
        while node.left is not None:
            node = node.left
        return node

    def _delete_node(self, node: Optional[_BSTNode], key: str) -> Optional[_BSTNode]:
        if node is None:
            return None
        if key < node.key:
            node.left = self._delete_node(node.left, key)
        elif key > node.key:
            node.right = self._delete_node(node.right, key)
        else:
            if node.left is None:
                self._size -= 1
                return node.right
            elif node.right is None:
                self._size -= 1
                return node.left
            temp = self._min_node(node.right)
            node.key = temp.key
            node.postings = temp.postings
            node.right = self._delete_node(node.right, temp.key)
        return node

    def _inorder(self, node: Optional[_BSTNode], result: list):
        if node is None:
            return
        self._inorder(node.left, result)
        result.append(node)
        self._inorder(node.right, result)

    # ---- InvertedIndexBase interface --------------------------------------
    def build(self, processed_docs: List[List[str]]) -> None:
        self._root = None
        self._size = 0
        # Build from sorted terms for a balanced-ish tree
        tmp: Dict[str, set] = defaultdict(set)
        for doc_id, tokens in enumerate(processed_docs):
            for token in tokens:
                tmp[token].add(doc_id)

        sorted_terms = sorted(tmp.keys())

        def _build_balanced(lo: int, hi: int):
            if lo > hi:
                return None
            mid = (lo + hi) // 2
            t = sorted_terms[mid]
            node = _BSTNode(t)
            node.postings = tmp[t]
            self._size += 1
            node.left = _build_balanced(lo, mid - 1)
            node.right = _build_balanced(mid + 1, hi)
            return node

        self._root = _build_balanced(0, len(sorted_terms) - 1)

    def search(self, term: str) -> List[int]:
        node = self._find_node(self._root, term)
        if node is None:
            return []
        return sorted(node.postings)

    def insert(self, term: str, doc_id: int) -> None:
        node = self._find_node(self._root, term)
        if node is not None:
            node.postings.add(doc_id)
        else:
            self._root = self._insert_node(self._root, term)
            new_node = self._find_node(self._root, term)
            new_node.postings.add(doc_id)

    def delete(self, term: str, doc_id: int) -> None:
        node = self._find_node(self._root, term)
        if node is None:
            return
        node.postings.discard(doc_id)
        if not node.postings:
            self._root = self._delete_node(self._root, term)

    def update(self, old_term: str, new_term: str, doc_id: int) -> None:
        self.delete(old_term, doc_id)
        self.insert(new_term, doc_id)

    def vocab_size(self) -> int:
        return self._size

    def memory_bytes(self) -> int:
        total = 0
        nodes = []
        self._inorder(self._root, nodes)
        for n in nodes:
            total += sys.getsizeof(n)
            total += sys.getsizeof(n.key)
            total += sys.getsizeof(n.postings)
            for d in n.postings:
                total += sys.getsizeof(d)
        return total

    def all_terms(self) -> List[str]:
        nodes = []
        self._inorder(self._root, nodes)
        return [n.key for n in nodes]

    def get_postings_set(self, term: str) -> set:
        node = self._find_node(self._root, term)
        if node is None:
            return set()
        return node.postings


# ============================================================================
# Benchmarking utilities
# ============================================================================
def benchmark_index(
    index: InvertedIndexBase,
    processed_docs: List[List[str]],
    sample_terms: List[str] = None,
    n_ops: int = 1000,
) -> dict:
    """
    Benchmark an inverted-index implementation.

    Returns a dict with timing and memory metrics.
    """
    # --- Build ---
    t0 = time.perf_counter()
    index.build(processed_docs)
    build_time = time.perf_counter() - t0

    mem = index.memory_bytes()

    if sample_terms is None:
        all_t = index.all_terms()
        step = max(1, len(all_t) // n_ops)
        sample_terms = all_t[::step][:n_ops]

    # --- Retrieval ---
    t0 = time.perf_counter()
    for term in sample_terms:
        index.search(term)
    retrieval_time = time.perf_counter() - t0

    # --- Insertion ---
    dummy_doc = len(processed_docs) + 999
    t0 = time.perf_counter()
    for term in sample_terms:
        index.insert(term, dummy_doc)
    insertion_time = time.perf_counter() - t0

    # --- Deletion ---
    t0 = time.perf_counter()
    for term in sample_terms:
        index.delete(term, dummy_doc)
    deletion_time = time.perf_counter() - t0

    # --- Update ---
    t0 = time.perf_counter()
    for i, term in enumerate(sample_terms):
        new_term = term + "_upd"
        index.update(term, new_term, 0)
        # revert
        index.update(new_term, term, 0)
    update_time = time.perf_counter() - t0

    num_ops = len(sample_terms)
    return {
        "name": index.name,
        "vocab_size": index.vocab_size(),
        "memory_bytes": mem,
        "memory_KB": round(mem / 1024, 2),
        "build_time_s": round(build_time, 6),
        "retrieval_total_s": round(retrieval_time, 6),
        "retrieval_avg_us": round(retrieval_time / num_ops * 1e6, 2),
        "insertion_total_s": round(insertion_time, 6),
        "insertion_avg_us": round(insertion_time / num_ops * 1e6, 2),
        "deletion_total_s": round(deletion_time, 6),
        "deletion_avg_us": round(deletion_time / num_ops * 1e6, 2),
        "update_total_s": round(update_time, 6),
        "update_avg_us": round(update_time / num_ops * 1e6, 2),
        "num_ops_benchmarked": num_ops,
    }
