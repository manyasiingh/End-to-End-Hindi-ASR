"""WER / CER with substitution-deletion-insertion breakdown and bootstrap CIs.

Implemented on top of rapidfuzz's Levenshtein (identical to the classic
word-level edit distance used by jiwer; tests/test_metrics.py checks this).
All inputs are expected to be ALREADY normalised (see textnorm.py).
"""
from __future__ import annotations

import numpy as np
from rapidfuzz.distance import Levenshtein


def _counts(ref_seq, hyp_seq):
    """Return (S, D, I) for one utterance. Sequences are str (chars) or list[str] (words)."""
    ops = Levenshtein.editops(ref_seq, hyp_seq)
    s = d = i = 0
    for op in ops:
        tag = op.tag
        if tag == "replace":
            s += 1
        elif tag == "delete":
            d += 1
        elif tag == "insert":
            i += 1
    return s, d, i


def per_utterance(refs: list[str], hyps: list[str], level: str = "word") -> np.ndarray:
    """-> int array (N, 4): S, D, I, ref_len. Utterances with an empty reference are dropped."""
    rows = []
    for r, h in zip(refs, hyps):
        if level == "word":
            rs, hs = r.split(), h.split()
        else:
            rs, hs = r, h  # characters, spaces included (same convention as jiwer.cer)
        if len(rs) == 0:
            continue
        s, d, i = _counts(rs, hs)
        rows.append((s, d, i, len(rs)))
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def rate(a: np.ndarray) -> float:
    n = a[:, 3].sum()
    return float((a[:, 0] + a[:, 1] + a[:, 2]).sum() / n) if n else float("nan")


def breakdown(a: np.ndarray) -> dict:
    n = max(int(a[:, 3].sum()), 1)
    return {
        "err": rate(a),
        "sub": float(a[:, 0].sum() / n),
        "del": float(a[:, 1].sum() / n),
        "ins": float(a[:, 2].sum() / n),
        "ref_len": int(a[:, 3].sum()),
    }


def bootstrap_ci(a: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Utterance-level bootstrap CI of the corpus-level error rate."""
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    errs = (a[:, 0] + a[:, 1] + a[:, 2]).astype(np.float64)
    lens = a[:, 3].astype(np.float64)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        out[b] = errs[idx].sum() / max(lens[idx].sum(), 1.0)
    return (float(np.percentile(out, 100 * alpha / 2)), float(np.percentile(out, 100 * (1 - alpha / 2))))


def wer_cer(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    return rate(per_utterance(refs, hyps, "word")), rate(per_utterance(refs, hyps, "char"))
