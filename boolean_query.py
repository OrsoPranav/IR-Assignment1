"""
Part (d): Boolean Query Processing on the Inverted Index
=========================================================
Supports AND, OR, NOT operators with a simple recursive-descent parser.

Also includes optimisation strategies:
  1. Naive evaluation
  2. Sorted merge-intersect
  3. Query term ordering (process smallest postings first)
"""

import re
import time
from typing import List, Set, Tuple

from inverted_index import InvertedIndexBase


# ============================================================================
# Boolean expression parser  (recursive descent)
# ============================================================================
# Grammar:
#   expr   -> term ((AND | OR) term)*
#   term   -> NOT term | LPAREN expr RPAREN | WORD
#
# Token types: WORD, AND, OR, NOT, LPAREN, RPAREN

_TOKEN_RE = re.compile(
    r"\s*(AND|OR|NOT|[()]|[A-Za-z0-9_]+)\s*", re.IGNORECASE
)


def _tokenize_query(query: str) -> List[str]:
    return [m.group(1) for m in _TOKEN_RE.finditer(query)]


class _Parser:
    """Recursive-descent parser for Boolean queries."""

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos].upper()
        return None

    def consume(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self):
        node = self.parse_expr()
        return node

    def parse_expr(self):
        left = self.parse_term()
        while self.peek() in ("AND", "OR"):
            op = self.consume().upper()
            right = self.parse_term()
            left = (op, left, right)
        return left

    def parse_term(self):
        if self.peek() == "NOT":
            self.consume()
            operand = self.parse_term()
            return ("NOT", operand)
        if self.peek() == "(":
            self.consume()
            node = self.parse_expr()
            if self.peek() == ")":
                self.consume()
            return node
        # it's a WORD
        return ("TERM", self.consume().lower())


def parse_boolean_query(query: str):
    """Parse a Boolean query string into an AST."""
    tokens = _tokenize_query(query)
    parser = _Parser(tokens)
    return parser.parse()


# ============================================================================
# Evaluation
# ============================================================================
def evaluate(
    ast,
    index: InvertedIndexBase,
    all_doc_ids: Set[int],
) -> Set[int]:
    """Evaluate a parsed Boolean AST against an inverted index."""
    if ast[0] == "TERM":
        return set(index.search(ast[1]))
    elif ast[0] == "NOT":
        child = evaluate(ast[1], index, all_doc_ids)
        return all_doc_ids - child
    elif ast[0] == "AND":
        left = evaluate(ast[1], index, all_doc_ids)
        right = evaluate(ast[2], index, all_doc_ids)
        return left & right
    elif ast[0] == "OR":
        left = evaluate(ast[1], index, all_doc_ids)
        right = evaluate(ast[2], index, all_doc_ids)
        return left | right
    else:
        raise ValueError(f"Unknown AST node: {ast[0]}")


def run_boolean_query(
    query_str: str,
    index: InvertedIndexBase,
    num_docs: int,
) -> Tuple[Set[int], float]:
    """
    Parse and evaluate a Boolean query.

    Returns (result_set, elapsed_seconds).
    """
    all_doc_ids = set(range(num_docs))
    ast = parse_boolean_query(query_str)
    t0 = time.perf_counter()
    result = evaluate(ast, index, all_doc_ids)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ============================================================================
# Optimised sorted-merge intersection
# ============================================================================
def sorted_intersect(a: List[int], b: List[int]) -> List[int]:
    """Merge-intersect two sorted posting lists in O(|a|+|b|)."""
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def sorted_union(a: List[int], b: List[int]) -> List[int]:
    """Merge-union two sorted posting lists in O(|a|+|b|)."""
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


def evaluate_optimized(
    ast,
    index: InvertedIndexBase,
    all_doc_ids_sorted: List[int],
) -> List[int]:
    """Evaluate using sorted merge operations (returns sorted list)."""
    if ast[0] == "TERM":
        return index.search(ast[1])  # already sorted
    elif ast[0] == "NOT":
        child = set(evaluate_optimized(ast[1], index, all_doc_ids_sorted))
        return [d for d in all_doc_ids_sorted if d not in child]
    elif ast[0] == "AND":
        left = evaluate_optimized(ast[1], index, all_doc_ids_sorted)
        right = evaluate_optimized(ast[2], index, all_doc_ids_sorted)
        return sorted_intersect(left, right)
    elif ast[0] == "OR":
        left = evaluate_optimized(ast[1], index, all_doc_ids_sorted)
        right = evaluate_optimized(ast[2], index, all_doc_ids_sorted)
        return sorted_union(left, right)
    else:
        raise ValueError(f"Unknown AST node: {ast[0]}")


def run_boolean_query_optimized(
    query_str: str,
    index: InvertedIndexBase,
    num_docs: int,
) -> Tuple[List[int], float]:
    """
    Parse and evaluate a Boolean query using sorted-merge optimization.
    """
    all_sorted = list(range(num_docs))
    ast = parse_boolean_query(query_str)
    t0 = time.perf_counter()
    result = evaluate_optimized(ast, index, all_sorted)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ============================================================================
# Query-term-ordering optimisation  (AND chains only)
# ============================================================================
def _collect_and_terms(ast) -> list:
    """Flatten an AND-chain into a list of leaf ASTs."""
    if ast[0] == "AND":
        return _collect_and_terms(ast[1]) + _collect_and_terms(ast[2])
    return [ast]


def evaluate_ordered(
    ast,
    index: InvertedIndexBase,
    all_doc_ids: Set[int],
) -> Set[int]:
    """
    Optimised evaluation that processes AND terms in ascending
    posting-list-size order (smallest first).
    """
    if ast[0] == "AND":
        terms = _collect_and_terms(ast)
        # sort by posting list length
        sized = []
        for t in terms:
            if t[0] == "TERM":
                sized.append((len(index.search(t[1])), t))
            else:
                sized.append((len(all_doc_ids), t))
        sized.sort(key=lambda x: x[0])

        result = evaluate_ordered(sized[0][1], index, all_doc_ids)
        for _, sub_ast in sized[1:]:
            result = result & evaluate_ordered(sub_ast, index, all_doc_ids)
            if not result:
                break  # short-circuit
        return result

    return evaluate(ast, index, all_doc_ids)


def run_boolean_query_ordered(
    query_str: str,
    index: InvertedIndexBase,
    num_docs: int,
) -> Tuple[Set[int], float]:
    all_doc_ids = set(range(num_docs))
    ast = parse_boolean_query(query_str)
    t0 = time.perf_counter()
    result = evaluate_ordered(ast, index, all_doc_ids)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ============================================================================
# Complexity analysis helper
# ============================================================================
def complexity_analysis() -> str:
    """Return a formatted string explaining the time complexity."""
    return """
Time Complexity Analysis for Boolean Query Evaluation
======================================================

Assume:
  - N  = number of distinct terms in the inverted index
  - P  = average length of a posting list
  - k  = number of query terms

Data Structure     | Lookup   | AND (2 terms) | OR (2 terms)
-----------------------------------------------------------
HashMap (dict)     | O(1)     | O(P₁ + P₂)   | O(P₁ + P₂)
SortedList+bisect  | O(log N) | O(P₁ + P₂)   | O(P₁ + P₂)
BST                | O(log N)*| O(P₁ + P₂)   | O(P₁ + P₂)

* BST worst-case is O(N) if unbalanced.

For the full query  term1 AND term2 AND term3:
  - Naive (set intersection):      O(P₁ + P₂ + P₃)
  - Sorted merge:                   O(P₁ + P₂ + P₃)
  - Ordered (smallest first):       O(P_min * k)  (short-circuits early)

For  term1 OR term2 AND NOT term3:
  - Parsed as: term1 OR (term2 AND (NOT term3))
  - Cost: O(|all_docs| + P₂ + P₃)  — NOT requires complement against corpus

Optimization Strategies:
  1. Process AND terms in ascending posting-list size ⇒ early termination.
  2. Use sorted posting lists + merge-intersect ⇒ avoid hashing overhead.
  3. Skip pointers / galloping search for large lists ⇒ O(P_small · log(P_large)).
  4. Caching frequent posting lists in memory.
"""
