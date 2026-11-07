"""Deterministic synthetic classification data with a real distribution shift.

Documents are short token sequences from class-conditional vocabularies.
Distribution A uses the base vocabulary; distribution B applies the shift a
real deployment sees: a new vocabulary for the same concepts (token remap) and
a new token order prior. Hashed bag-of-words features make the mapping exact
and reproducible - one seed reproduces both distributions bit for bit.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Sequence, Tuple

_CLASSES = ["sports", "tech", "food"]
_BASE_VOCAB = {
    "sports": ["goal", "match", "league", "score", "coach", "tournament"],
    "tech": ["kernel", "compiler", "server", "protocol", "debugger", "cache"],
    "food": ["recipe", "sauce", "baked", "flavor", "kitchen", "spice"],
}
# Shifted vocabulary: the same concepts under new surface tokens (deployment
# drift), plus remapped class emphases.
_SHIFT_VOCAB = {
    "sports": ["touchdown", "playoff", "stadium", "referee", "athlete", "championship"],
    "tech": ["microservice", "latency", "deploy", "kubernetes", "endpoint", "rollback"],
    "food": ["umami", "ferment", "glaze", "sousvide", "pantry", "marinade"],
}
_FILLER = ["the", "a", "of", "in", "and", "new", "big", "local"]


def generate_split(distribution: str, n_per_class: int, seed: int = 0,
                   doc_len: int = 8) -> List[Tuple[List[str], int]]:
    """Deterministic documents: (tokens, class_index) pairs."""
    if distribution not in ("A", "B"):
        raise ValueError("distribution must be 'A' or 'B'")
    vocab = _BASE_VOCAB if distribution == "A" else _SHIFT_VOCAB
    rng = random.Random(seed + (0 if distribution == "A" else 1000))
    docs: List[Tuple[List[str], int]] = []
    for ci, cls in enumerate(_CLASSES):
        for _ in range(n_per_class):
            n_signal = rng.randint(3, 5)
            toks = [rng.choice(vocab[cls]) for _ in range(n_signal)]
            toks += [rng.choice(_FILLER) for _ in range(doc_len - n_signal)]
            rng.shuffle(toks)
            docs.append((toks, ci))
    rng.shuffle(docs)
    return docs


def classes() -> List[str]:
    return list(_CLASSES)


def featurize(tokens: Sequence[str], dim: int) -> List[float]:
    """Hashed bag-of-words: exact, deterministic, dependency-free."""
    vec = [0.0] * dim
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = sum(vec) or 1.0
    return [v / norm for v in vec]
