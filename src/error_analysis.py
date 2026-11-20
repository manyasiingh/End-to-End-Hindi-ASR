#!/usr/bin/env python
"""Error analysis: WHERE the word/character errors come from, per model and test set.

Reads only saved predictions (results*/<model>/<set>.jsonl, already normalised by textnorm) and, optionally,
the training manifest (for out-of-vocabulary analysis). No model, no GPU. Every number in the report is
computed here and also written to <out>.json so it can be traced.

What it measures
  1. Overview         WER / CER / lenient WER, S-D-I, hypothesis-length ratio.
  2. Concentration    how many utterances are perfect, how many have WER >= 100 %, and how much of the total
                      error comes from the worst 5 % of utterances (a long tail = a few catastrophic outputs).
  3. Error sources    every word error is attributed to exactly one cause:
                        utterance-level (whole utterance):  empty output | loop / hallucination | wrong script
                        word-level (WER alignment):         word boundary (split/merge) | nukta/chandrabindu only |
                                                            matra/diacritic only | spelling near-miss | number format |
                                                            non-Devanagari word | wrong word | deleted words |
                                                            inserted words
  4. Characters       which Devanagari character classes the character errors hit (matras, halant, nasals...),
                      and the most frequent character confusions.
  5. Vocabulary       out-of-vocabulary rate of each test set w.r.t. the training transcripts, and the error rate
                      on OOV vs in-vocabulary reference words.
  6. Duration         WER by utterance duration bucket.
  7. Pairwise         per-utterance wins, oracle WER (complementarity), WER on utterances where neither model
                      loops, and a per-category decomposition of the error-count gap between two models.

Usage
  python src/error_analysis.py --results results_quick --train_manifest data_quick/kathbath_train/manifest.jsonl \
      --pair whisper_small_finetuned,mms300m_ctc_finetuned --pair whisper_small_finetuned,whisper_small_zeroshot
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textnorm import normalize  # noqa: E402

# ----------------------------------------------------------------------------------------------------------
# Devanagari character classes
# ----------------------------------------------------------------------------------------------------------
NUKTA, HALANT = "़", "्"
NASALS = {"ँ", "ं", "ः"}                       # chandrabindu, anusvara, visarga
MATRAS = {chr(c) for c in range(0x093A, 0x094D) if chr(c) != NUKTA} | {"ॕ", "ॖ", "ॗ",
                                                                       "ॢ", "ॣ"}
INDEP_VOWELS = {chr(c) for c in range(0x0904, 0x0915)} | {"ॠ", "ॡ"} | \
               {chr(c) for c in range(0x0972, 0x0978)}
DIACRITICS = MATRAS | NASALS | {NUKTA, HALANT}


def char_class(ch: str) -> str:
    if ch == " ":
        return "space (word boundary)"
    if ch in MATRAS:
        return "vowel sign (matra)"
    if ch == HALANT:
        return "halant (virama)"
    if ch in NASALS:
        return "anusvara/chandrabindu/visarga"
    if ch == NUKTA:
        return "nukta"
    if ch in INDEP_VOWELS:
        return "independent vowel"
    if "ऀ" <= ch <= "ॿ":
        return "consonant"
    if ch.isdigit():
        return "digit"
    if ch.isascii() and ch.isalpha():
        return "Latin letter"
    return "other"


def script_of(ch: str) -> str | None:
    """Script of a letter/mark; None for spaces, digits, etc."""
    if not (ch.isalpha() or unicodedata.category(ch).startswith("M")):
        return None
    if "ऀ" <= ch <= "ॿ" or "꣠" <= ch <= "ꣿ":
        return "Devanagari"
    if ch.isascii():
        return "Latin"
    if "؀" <= ch <= "ۿ" or "ݐ" <= ch <= "ݿ" or "ﭐ" <= ch <= "﻿":
        return "Arabic/Urdu"
    return "other"


def skeleton(word: str) -> str:
    """Word with all Devanagari diacritics removed (consonants + independent vowels only)."""
    w = unicodedata.normalize("NFD", word)
    return "".join(c for c in w if c not in DIACRITICS)


# ----------------------------------------------------------------------------------------------------------
# Utterance-level failure flags
# ----------------------------------------------------------------------------------------------------------
def max_repeat(words: list[str], max_n: int = 4) -> int:
    """Largest number of consecutive repetitions of any n-gram (n <= max_n)."""
    best = 1
    for n in range(1, max_n + 1):
        i = 0
        while i + 2 * n <= len(words):
            k = 1
            while i + (k + 1) * n <= len(words) and words[i + k * n:i + (k + 1) * n] == words[i:i + n]:
                k += 1
            best = max(best, k)
            i += max(1, (k - 1) * n) if k > 1 else 1
    return best


def is_loop(ref: list[str], hyp: list[str]) -> bool:
    """Hallucination / repetition loop: far longer than the reference, or an n-gram repeated >= 4 times in a row
    (while the reference has no such repetition)."""
    if len(hyp) > 1.5 * len(ref) + 3:
        return True
    return max_repeat(hyp) >= 4 and max_repeat(ref) < 4


def foreign_share(text: str) -> float:
    scripts = [s for s in map(script_of, text) if s]
    return 0.0 if not scripts else sum(s != "Devanagari" for s in scripts) / len(scripts)


# ----------------------------------------------------------------------------------------------------------
# Alignment-based error attribution
# ----------------------------------------------------------------------------------------------------------
UTT_CATS = ["empty output", "loop / hallucination", "wrong script (>30 % non-Devanagari)"]
REGION_CATS = ["word boundary (split/merge)", "nukta/chandrabindu only", "matra/diacritic only",
               "spelling near-miss", "number format", "non-Devanagari word", "wrong word",
               "deleted words", "inserted words"]
ALL_CATS = UTT_CATS + REGION_CATS


def error_regions(ref: list[str], hyp: list[str]):
    """Maximal non-matching spans of the optimal word alignment: list of (ref_span, hyp_span)."""
    out, cur_r, cur_h = [], [], []
    for op in Levenshtein.opcodes(ref, hyp):
        if op.tag == "equal":
            if cur_r or cur_h:
                out.append((cur_r, cur_h))
                cur_r, cur_h = [], []
        else:
            cur_r += ref[op.src_start:op.src_end]
            cur_h += hyp[op.dest_start:op.dest_end]
    if cur_r or cur_h:
        out.append((cur_r, cur_h))
    return out


def classify_pair(r: str, h: str) -> str:
    if any(c.isdigit() for c in r + h):
        return "number format"
    if foreign_share(h) > 0.5:
        return "non-Devanagari word"
    if normalize(r, lenient=True) == normalize(h, lenient=True):
        return "nukta/chandrabindu only"
    if skeleton(r) == skeleton(h):
        return "matra/diacritic only"
    if Levenshtein.normalized_similarity(r, h) >= 0.6:
        return "spelling near-miss"
    return "wrong word"


def _joins(a: str, b: str) -> bool:
    """a == b up to a small spelling difference (used to recognise merged / split words)."""
    return a == b or (len(b) >= 4 and Levenshtein.normalized_similarity(a, b) >= 0.8)


def attribute(ref: list[str], hyp: list[str]) -> Counter:
    """Split this utterance's word errors (= word edit distance) into categories.

    Within each error span the standard WER alignment (edit ops) is used, so substitutions, deletions and
    insertions are counted exactly as in the WER. A deletion next to a substitution whose hypothesis word is
    the two reference words written together (merge), or an insertion next to a substitution whose two
    hypothesis words form the reference word (split), is a word-boundary error instead (2 errors).
    """
    c = Counter()
    for r, h in error_regions(ref, hyp):
        cost = Levenshtein.distance(r, h)
        if "".join(r) == "".join(h):
            c["word boundary (split/merge)"] += cost
            continue
        ops = Levenshtein.editops(r, h)
        rep_r = {o.src_pos: o.dest_pos for o in ops if o.tag == "replace"}
        rep_h = {o.dest_pos: o.src_pos for o in ops if o.tag == "replace"}
        dels = [o.src_pos for o in ops if o.tag == "delete"]
        ins = [o.dest_pos for o in ops if o.tag == "insert"]
        used_r, used_h = set(), set()
        for i in dels:                       # merge: two ref words -> one hyp word
            for k, first, second in ((i - 1, i - 1, i), (i + 1, i, i + 1)):
                if k in rep_r and k not in used_r and _joins(r[first] + r[second], h[rep_r[k]]):
                    c["word boundary (split/merge)"] += 2
                    used_r.update((i, k))
                    break
        for j in ins:                        # split: one ref word -> two hyp words
            for k, first, second in ((j - 1, j - 1, j), (j + 1, j, j + 1)):
                if k in rep_h and rep_h[k] not in used_r and _joins(h[first] + h[second], r[rep_h[k]]):
                    c["word boundary (split/merge)"] += 2
                    used_h.add(j)
                    used_r.add(rep_h[k])
                    break
        for i, j in rep_r.items():
            if i not in used_r:
                c[classify_pair(r[i], h[j])] += 1
        c["deleted words"] += sum(i not in used_r for i in dels)
        c["inserted words"] += sum(j not in used_h for j in ins)
        assert sum(o.tag != "equal" for o in ops) == cost
    return c


# ----------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------
def load_results(root: Path):
    models = sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.jsonl")))
    roles = {}
    summ = root / "summary.csv"
    if summ.exists():
        for row in csv.DictReader(open(summ, encoding="utf-8")):
            roles[row["set"]] = row["role"]
    sets = sorted({p.stem for m in models for p in (root / m).glob("*.jsonl")},
                  key=lambda s: (roles.get(s, "ood") != "in_domain", s))
    data = {}
    for m in models:
        for s in sets:
            f = root / m / f"{s}.jsonl"
            if f.exists():
                data[(m, s)] = [json.loads(line) for line in open(f, encoding="utf-8")]
    return models, sets, roles, data


def pct(x, n):
    return 100.0 * x / n if n else float("nan")


# ----------------------------------------------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------------------------------------------
def analyse_pair(recs, train_vocab):
    n_ref_w = n_ref_c = 0
    S = D = I = 0
    c_err = len_err = 0
    hyp_w = 0
    per_utt = []                       # (errors, ref_len, duration, id)
    cats = Counter()
    flags = Counter()
    char_cls = Counter()
    confusions = Counter()
    script = Counter()
    oov_tok = oov_err = iv_tok = iv_err = 0
    for j in recs:
        ref, hyp = j["ref"].split(), j["hyp"].split()
        if not ref:
            continue
        n_ref_w += len(ref)
        n_ref_c += len(j["ref"])
        hyp_w += len(hyp)
        ops = Levenshtein.editops(ref, hyp)
        s = sum(o.tag == "replace" for o in ops)
        d = sum(o.tag == "delete" for o in ops)
        i = sum(o.tag == "insert" for o in ops)
        S, D, I = S + s, D + d, I + i
        err = s + d + i
        c_err += Levenshtein.distance(j["ref"], j["hyp"])
        len_err += Levenshtein.distance(normalize(j["ref"], lenient=True).split(),
                                        normalize(j["hyp"], lenient=True).split())
        per_utt.append((err, len(ref), j.get("duration", float("nan")), j["id"], bool(hyp) and is_loop(ref, hyp)))
        for ch in j["hyp"]:
            sc = script_of(ch)
            if sc:
                script[sc] += 1

        # whole-utterance failure modes take precedence
        if not hyp:
            flags["empty output"] += 1
            cats["empty output"] += err
        elif is_loop(ref, hyp):
            flags["loop / hallucination"] += 1
            cats["loop / hallucination"] += err
        elif foreign_share(j["hyp"]) > 0.3:
            flags["wrong script (>30 % non-Devanagari)"] += 1
            cats["wrong script (>30 % non-Devanagari)"] += err
        else:
            cats.update(attribute(ref, hyp))
            # character-level detail only for "ordinary" utterances (loops would swamp it)
            for o in Levenshtein.editops(j["ref"], j["hyp"]):
                if o.tag == "replace":
                    a, b = j["ref"][o.src_pos], j["hyp"][o.dest_pos]
                    char_cls[char_class(a)] += 1
                    confusions[(a, b)] += 1
                elif o.tag == "delete":
                    a = j["ref"][o.src_pos]
                    char_cls[char_class(a)] += 1
                    confusions[(a, "∅")] += 1
                else:
                    b = j["hyp"][o.dest_pos]
                    char_cls[char_class(b)] += 1
                    confusions[("∅", b)] += 1

        # OOV: was each reference word recognised (inside an 'equal' block)?
        if train_vocab is not None:
            ok = [False] * len(ref)
            for op in Levenshtein.opcodes(ref, hyp):
                if op.tag == "equal":
                    for k in range(op.src_start, op.src_end):
                        ok[k] = True
            for w, good in zip(ref, ok):
                if w in train_vocab:
                    iv_tok += 1
                    iv_err += not good
                else:
                    oov_tok += 1
                    oov_err += not good

    errs = np.array([u[0] for u in per_utt], dtype=float)
    lens = np.array([u[1] for u in per_utt], dtype=float)
    utt_wer = errs / np.maximum(lens, 1)
    total = errs.sum()
    k = max(1, int(round(0.05 * len(errs))))
    worst5 = np.sort(errs)[::-1][:k].sum()
    assert sum(cats.values()) == total, "error attribution must account for every word error"
    return {
        "n_utts": len(per_utt), "ref_words": n_ref_w, "ref_chars": n_ref_c,
        "wer": pct(S + D + I, n_ref_w), "cer": pct(c_err, n_ref_c), "wer_lenient": pct(len_err, n_ref_w),
        "sub": pct(S, n_ref_w), "del": pct(D, n_ref_w), "ins": pct(I, n_ref_w),
        "hyp_len_ratio": hyp_w / max(n_ref_w, 1),
        "perfect_utts_pct": pct(int((errs == 0).sum()), len(errs)),
        "wer_ge_100_utts_pct": pct(int((utt_wer >= 1).sum()), len(errs)),
        "median_utt_wer": float(np.median(utt_wer) * 100),
        "worst5pct_error_share": pct(worst5, total),
        "total_word_errors": int(total),
        "categories": {c: int(cats.get(c, 0)) for c in ALL_CATS},
        "flags": {c: int(flags.get(c, 0)) for c in UTT_CATS},
        "char_error_classes": dict(char_cls.most_common()),
        "top_confusions": [[a, b, n] for (a, b), n in confusions.most_common(12)],
        "hyp_script_share": {k: pct(v, sum(script.values())) for k, v in script.most_common()},
        "oov": None if train_vocab is None else {
            "ref_oov_rate": pct(oov_tok, oov_tok + iv_tok),
            "oov_word_err": pct(oov_err, oov_tok), "iv_word_err": pct(iv_err, iv_tok),
            "oov_tokens": oov_tok, "iv_tokens": iv_tok},
        "_per_utt": per_utt,
    }


DUR_BUCKETS = [(0, 3), (3, 6), (6, 10), (10, 20), (20, 31)]


def duration_table(per_utt):
    out = {}
    for lo, hi in DUR_BUCKETS:
        sel = [(u[0], u[1]) for u in per_utt if lo <= u[2] < hi]
        if sel:
            out[f"{lo}-{hi}s"] = (len(sel), pct(sum(e for e, _ in sel), sum(n for _, n in sel)))
    return out


def pairwise(ra, rb):
    """Two models' analyses on the same set (utterances matched by id)."""
    A = {u[3]: u for u in ra["_per_utt"]}
    B = {u[3]: u for u in rb["_per_utt"]}
    ids = [i for i in A if i in B]
    wa = sum(A[i][0] < B[i][0] for i in ids)
    wb = sum(A[i][0] > B[i][0] for i in ids)
    n = sum(A[i][1] for i in ids)
    oracle = sum(min(A[i][0], B[i][0]) for i in ids)
    clean = [i for i in ids if not A[i][4] and not B[i][4]]           # neither model loops
    nc = sum(A[i][1] for i in clean)
    return {"a_better": wa, "b_better": wb, "tie": len(ids) - wa - wb, "oracle_wer": pct(oracle, n),
            "no_loop_utts": len(clean), "a_wer_no_loop": pct(sum(A[i][0] for i in clean), nc),
            "b_wer_no_loop": pct(sum(B[i][0] for i in clean), nc),
            # error-count difference B - A per category: which causes explain the gap
            "gap_by_category": {c: rb["categories"][c] - ra["categories"][c] for c in ALL_CATS},
            "gap_total": rb["total_word_errors"] - ra["total_word_errors"]}


# ----------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------
def fmt(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def make_chart(res, models, sets, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # reference categorical palette (dataviz skill, light mode), fixed order; muted gray for the remainder
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    groups = {  # 8 legend groups, so each gets one palette slot
        "loop / hallucination": ["loop / hallucination"],
        "empty / wrong script": ["empty output", "wrong script (>30 % non-Devanagari)", "non-Devanagari word"],
        "word boundary": ["word boundary (split/merge)"],
        "diacritics (nukta, matra)": ["nukta/chandrabindu only", "matra/diacritic only"],
        "spelling near-miss": ["spelling near-miss", "number format"],
        "wrong word": ["wrong word"],
        "deleted words": ["deleted words"],
        "inserted words": ["inserted words"],
    }
    fig, axes = plt.subplots(len(sets), 1, figsize=(10, 1.15 * len(models) * len(sets) + 1.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, sets):
        ylabels = []
        for yi, m in enumerate(models):
            r = res.get((m, s))
            ylabels.append(m)
            if not r:
                continue
            left = 0.0
            for gi, (g, members) in enumerate(groups.items()):
                v = 100.0 * sum(r["categories"][c] for c in members) / r["ref_words"]
                if v > 0:
                    ax.barh(yi, v, left=left, color=pal[gi], height=0.62, edgecolor="#fcfcfb", linewidth=2)
                left += v
            ax.text(left + 1.5, yi, f"{r['wer']:.1f}", va="center", fontsize=8.5, color="#333333")
        ax.set_yticks(range(len(models)), ylabels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(s, loc="left", fontsize=10, color="#1a1a19")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("word errors as % of reference words (bar length = WER %)", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=pal[i]) for i in range(len(groups))]
    fig.legend(handles, list(groups), loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("Where the word errors come from", x=0.01, ha="left", fontsize=12, color="#1a1a19")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.patch.set_facecolor("#fcfcfb")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(res, models, sets, roles, pairs, out_md: Path, chart_name: str, results_dir: str):
    L = []
    w = L.append
    w("# Error analysis — where the errors come from\n")
    w(f"Generated by `src/error_analysis.py` from `{results_dir}/<model>/<set>.jsonl` (strictly normalised text). "
      "Every number is also in the companion `.json`. Categories are measured; any explanation of *why* they "
      "occur is a hypothesis unless measured separately.\n")

    w("## 1. Overview\n")
    w("| model | set | WER | CER | lenient WER | sub | del | ins | hyp/ref words |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in sets:
        for m in models:
            r = res.get((m, s))
            if r:
                w(f"| {m} | {s} | {fmt(r['wer'])} | {fmt(r['cer'])} | {fmt(r['wer_lenient'])} | {fmt(r['sub'])} | "
                  f"{fmt(r['del'])} | {fmt(r['ins'])} | {fmt(r['hyp_len_ratio'], 2)} |")
    w("\n*hyp/ref words* > 1 means the model outputs more words than were spoken (insertions / hallucination); "
      "< 1 means it drops words.\n")

    w("## 2. Error concentration\n")
    w("| model | set | perfect utts % | utts with WER ≥ 100 % | median utt WER | share of all errors in worst 5 % of utts |")
    w("|---|---|---|---|---|---|")
    for s in sets:
        for m in models:
            r = res.get((m, s))
            if r:
                w(f"| {m} | {s} | {fmt(r['perfect_utts_pct'])} | {fmt(r['wer_ge_100_utts_pct'])} % | "
                  f"{fmt(r['median_utt_wer'])} | {fmt(r['worst5pct_error_share'])} % |")
    w("\nA high worst-5 % share with a moderate median means the corpus WER is driven by a few catastrophic "
      "utterances, not by uniformly mediocre ones.\n")

    w("## 3. Error sources (share of each model's word errors)\n")
    w(f"![error sources]({chart_name})\n")
    w("Each word error is assigned to exactly one category (they sum to 100 %). Utterance-level categories "
      "(empty output, loop / hallucination, wrong script) take the whole utterance's errors; the rest come from "
      "aligned error spans.\n")
    for s in sets:
        w(f"### {s} ({roles.get(s, '?')})\n")
        w("| category | " + " | ".join(models) + " |")
        w("|---|" + "---|" * len(models))
        for c in ALL_CATS:
            vals = []
            for m in models:
                r = res.get((m, s))
                vals.append(fmt(pct(r["categories"][c], r["total_word_errors"])) if r else "—")
            if any(v not in ("0.0", "—") for v in vals):
                w(f"| {c} | " + " | ".join(vals) + " |")
        w("| **total word errors** | " + " | ".join(
            str(res[(m, s)]["total_word_errors"]) if (m, s) in res else "—" for m in models) + " |")
        w("| utterances flagged loop / empty / wrong script | " + " | ".join(
            "{} / {} / {}".format(res[(m, s)]["flags"]["loop / hallucination"], res[(m, s)]["flags"]["empty output"],
                                  res[(m, s)]["flags"]["wrong script (>30 % non-Devanagari)"])
            if (m, s) in res else "—" for m in models) + " |\n")
    w("Definitions: *word boundary* = words merged or split: same letters with different spaces "
      "(मध्यपूर्व ↔ मध्य पूर्व), allowing a small spelling difference (similarity ≥ 0.8) in the joined form; "
      "*nukta/chandrabindu only* = identical under the lenient normaliser; *matra/diacritic only* = same consonants "
      "and independent vowels, differing only in vowel signs, halant, nasal marks or nukta; *spelling near-miss* = "
      "character similarity ≥ 0.6; *number format* = digits involved; *wrong word* = any other substitution; "
      "*deleted / inserted words* = WER deletions / insertions not explained by a split or merge; "
      "*loop / hallucination* = hypothesis longer than 1.5 × reference + 3 words, or an n-gram repeated ≥ 4 "
      "times in a row. Substitutions, deletions and insertions follow the same alignment as the WER.\n")

    w("## 4. Character-level errors (utterances without loops / empty / wrong-script output)\n")
    classes = ["vowel sign (matra)", "consonant", "space (word boundary)", "halant (virama)",
               "anusvara/chandrabindu/visarga", "nukta", "independent vowel", "digit", "Latin letter", "other"]
    w("Share of character edits by the class of the affected character:\n")
    w("| model | set | " + " | ".join(classes) + " |")
    w("|---|---|" + "---|" * len(classes))
    for s in sets:
        for m in models:
            r = res.get((m, s))
            if r:
                tot = sum(r["char_error_classes"].values())
                w(f"| {m} | {s} | " + " | ".join(fmt(pct(r["char_error_classes"].get(c, 0), tot)) for c in classes) + " |")
    w("\nMost frequent character confusions (ref → hyp; ∅ = nothing), in-domain set:\n")
    s0 = sets[0]
    for m in models:
        r = res.get((m, s0))
        if r:
            w(f"* **{m}**: " + ", ".join(f"`{a}`→`{b}` ({n})" for a, b, n in r["top_confusions"][:10]))
    w("\nScript of the hypothesis letters (% of letters):\n")
    w("| model | set | Devanagari | Latin | Arabic/Urdu | other |")
    w("|---|---|---|---|---|---|")
    for s in sets:
        for m in models:
            r = res.get((m, s))
            if r:
                sh = r["hyp_script_share"]
                w(f"| {m} | {s} | " + " | ".join(fmt(sh.get(k, 0.0), 2) for k in
                                               ("Devanagari", "Latin", "Arabic/Urdu", "other")) + " |")

    if any(r["oov"] for r in res.values()):
        w("\n## 5. Vocabulary: out-of-vocabulary words (w.r.t. the training transcripts)\n")
        w("| set | OOV rate of reference tokens | " + " | ".join(f"{m}: IV err / OOV err" for m in models) + " |")
        w("|---|---|" + "---|" * len(models))
        for s in sets:
            any_r = next((res[(m, s)] for m in models if (m, s) in res), None)
            cells = []
            for m in models:
                o = res[(m, s)]["oov"] if (m, s) in res else None
                cells.append(f"{fmt(o['iv_word_err'])} / {fmt(o['oov_word_err'])}" if o else "—")
            w(f"| {s} | {fmt(any_r['oov']['ref_oov_rate'])} % | " + " | ".join(cells) + " |")
        w("\n*err* = % of reference words of that kind that were not recognised exactly (substituted or deleted). "
          "A large OOV − IV gap means the model struggles with words it never saw in training.\n")

    w("## 6. WER by utterance duration\n")
    buckets = [f"{lo}-{hi}s" for lo, hi in DUR_BUCKETS]
    w("| model | set | " + " | ".join(buckets) + " |")
    w("|---|---|" + "---|" * len(buckets))
    for s in sets:
        for m in models:
            r = res.get((m, s))
            if r:
                dt = duration_table(r["_per_utt"])
                w(f"| {m} | {s} | " + " | ".join(
                    f"{fmt(dt[b][1])} (n={dt[b][0]})" if b in dt else "—" for b in buckets) + " |")

    if pairs:
        w("\n## 7. Pairwise comparison (per utterance)\n")
        w("| set | A | B | A better | B better | tie | oracle WER (best of A/B per utt) |")
        w("|---|---|---|---|---|---|---|")
        for (s, a, b), p in pairs.items():
            w(f"| {s} | {a} | {b} | {p['a_better']} | {p['b_better']} | {p['tie']} | {fmt(p['oracle_wer'])} |")
        w("\nThe oracle WER is an upper bound on what combining the two systems could gain; it is not achievable "
          "without knowing which one is right.\n")
        w("**Without loops:** WER of both models on the utterances where *neither* produced a loop / hallucination:\n")
        w("| set | A | B | utts kept | A WER | B WER |")
        w("|---|---|---|---|---|---|")
        for (s, a, b), p in pairs.items():
            w(f"| {s} | {a} | {b} | {p['no_loop_utts']} | {fmt(p['a_wer_no_loop'])} | {fmt(p['b_wer_no_loop'])} |")
        w("\n**Gap decomposition:** word-error count of B minus A, per category (positive = B makes more errors of "
          "that kind). The rows sum to the total difference.\n")
        for a, b in dict.fromkeys((a, b) for _, a, b in pairs):
            keys = [k for k in pairs if k[1:] == (a, b)]
            w(f"*{b}* − *{a}*:\n")
            w("| category | " + " | ".join(k[0] for k in keys) + " |")
            w("|---|" + "---|" * len(keys))
            for c in ALL_CATS:
                vals = [pairs[k]["gap_by_category"][c] for k in keys]
                if any(vals):
                    w(f"| {c} | " + " | ".join(f"{v:+d}" for v in vals) + " |")
            w("| **total** | " + " | ".join(f"**{pairs[k]['gap_total']:+d}**" for k in keys) + " |\n")
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--train_manifest", default=None, help="training manifest.jsonl (enables OOV analysis)")
    ap.add_argument("--out", default=None, help="report path (default <results>/error_analysis.md)")
    ap.add_argument("--pair", action="append", default=None,
                    help="MODEL_A,MODEL_B for the pairwise table (repeatable; default: all fine-tuned pairs)")
    args = ap.parse_args()

    root = Path(args.results)
    out_md = Path(args.out) if args.out else root / "error_analysis.md"
    models, sets, roles, data = load_results(root)
    if not data:
        raise SystemExit(f"no predictions found under {root}")
    vocab = None
    if args.train_manifest:
        vocab = set()
        for line in open(args.train_manifest, encoding="utf-8"):
            vocab.update(json.loads(line)["text"].split())
        print(f"[error_analysis] training vocabulary: {len(vocab)} word types", flush=True)

    res = {}
    for (m, s), recs in data.items():
        res[(m, s)] = analyse_pair(recs, vocab)
        print(f"[error_analysis] {m} | {s}: WER {res[(m, s)]['wer']:.1f}", flush=True)

    if args.pair:
        pair_list = [tuple(p.split(",")) for p in args.pair]
    else:
        ft = [m for m in models if "zeroshot" not in m]
        pair_list = [(a, b) for i, a in enumerate(ft) for b in ft[i + 1:]]
    pairs = {}
    for a, b in pair_list:
        for s in sets:
            if (a, s) in res and (b, s) in res:
                pairs[(s, a, b)] = pairwise(res[(a, s)], res[(b, s)])

    chart = out_md.with_name("error_sources.png")
    make_chart(res, models, sets, chart)
    write_report(res, models, sets, roles, pairs, out_md, chart.name, root.as_posix())
    dump = {f"{m}|{s}": {k: v for k, v in r.items() if not k.startswith("_")} for (m, s), r in res.items()}
    dump["_pairwise"] = {f"{s}|{a}|{b}": p for (s, a, b), p in pairs.items()}
    out_md.with_suffix(".json").write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[error_analysis] wrote {out_md}, {out_md.with_suffix('.json')}, {chart}", flush=True)


if __name__ == "__main__":
    main()
