"""Reciprocal Rank Fusion — merging rankings that do not share a scale.

The vector branch returns cosine distances (0-2, lower is better); the lexical
branch returns ``ts_rank`` values (unbounded-ish, higher is better, and wildly
document-length dependent). Normalising either into the other's scale means
inventing a calibration that the data does not support, and it breaks the moment
the corpus changes.

RRF sidesteps the problem: it throws the scores away and keeps only the
POSITION. Each document scores ``1 / (k + rank)`` in every list it appears in,
and the fused score is the sum. Consequences worth stating out loud:

* A document ranked 3rd in BOTH branches beats one ranked 1st in a single branch
  — RRF rewards consensus, not peak performance in one channel.
* ``k`` (60 in the original Cormack et al. 2009 paper) flattens the curve near
  the top: without it, rank 1 would be worth twice rank 2 and a single branch
  could dictate the whole fusion.
* It is parameter-free beyond ``k``: nothing to train, nothing to re-tune when
  the embedding model changes.

Pure function, no I/O, no ORM types: the fusion is the part worth unit-testing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], *, k: int
) -> list[tuple[Hashable, float]]:
    """Fuse ranked id lists into one ranking, best first.

    Parameters
    ----------
    rankings:
        One sequence of ids per branch, already ordered best-first. Ids absent
        from a branch simply score nothing there — no penalty, no imputation.
    k:
        Smoothing constant. Higher = flatter, later ranks matter relatively more.

    Returns
    -------
    list[tuple[Hashable, float]]
        ``(id, fused_score)`` sorted by descending score. Ties break on the id so
        two runs over the same data produce the same ranking (the measurement
        table would be noise otherwise).
    """
    scores: dict[Hashable, float] = defaultdict(float)
    for ranking in rankings:
        for position, key in enumerate(ranking, start=1):
            scores[key] += 1.0 / (k + position)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
