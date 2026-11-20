#!/usr/bin/env python
"""Turn raw predictions into the results tables + chart.

Outputs (in --results dir): summary.csv, summary.md, errors_breakdown.md, wer_chart.png
Metrics: WER/CER on strictly-normalised text (primary) and WER on lenient text (nukta /
chandrabindu folded), utterance-level bootstrap 95% CI, S/D/I breakdown, RTF.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_meta  # noqa: E402
from metrics import bootstrap_ci, breakdown, per_utterance  # noqa: E402
from textnorm import normalize  # noqa: E402


def load_pairs(p: Path):
    refs, hyps, hyps_raw = [], [], []
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        refs.append(r["ref"])
        hyps.append(r["hyp"])
        hyps_raw.append(r["hyp_raw"])
    return refs, hyps, hyps_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--order", nargs="*", default=None, help="model display order")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    res = Path(args.results)
    models = sorted(d.name for d in res.iterdir() if d.is_dir())
    if args.order:
        models = [m for m in args.order if m in models] + [m for m in models if m not in args.order]
    rows = []
    for m in models:
        for p in sorted((res / m).glob("*.jsonl")):
            s = p.stem
            refs, hyps, hyps_raw = load_pairs(p)
            w = per_utterance(refs, hyps, "word")
            c = per_utterance(refs, hyps, "char")
            lo, hi = bootstrap_ci(w, args.boot)
            wl = per_utterance([normalize(r, True) for r in refs], [normalize(h, True) for h in hyps_raw], "word")
            bd = breakdown(w)
            timing = {}
            tp = res / m / f"{s}.timing.json"
            if tp.exists():
                timing = json.loads(tp.read_text())
            rows.append({
                "model": m, "set": s, "role": read_meta(Path(args.data) / s).get("role", "?"),
                "n_utts": len(w), "ref_words": bd["ref_len"],
                "wer": bd["err"], "wer_ci_lo": lo, "wer_ci_hi": hi,
                "cer": breakdown(c)["err"], "wer_lenient": breakdown(wl)["err"],
                "sub": bd["sub"], "del": bd["del"], "ins": bd["ins"],
                "rtf": timing.get("rtf", float("nan")),
                "empty_hyp_pct": 100.0 * sum(1 for h in hyps if not h.strip()) / max(len(hyps), 1),
            })
    if not rows:
        raise SystemExit("no prediction files found")
    fields = list(rows[0].keys())
    with open(res / "summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    sets = sorted({r["set"] for r in rows}, key=lambda s: (next(r["role"] for r in rows if r["set"] == s) != "in_domain", s))
    role_of = {r["set"]: r["role"] for r in rows}
    idx = {(r["model"], r["set"]): r for r in rows}

    lines = ["# Results (WER %, strict normalisation; 95% bootstrap CI in brackets)\n",
             "| model | " + " | ".join(f"{s} ({role_of[s]})" for s in sets) + " |",
             "|---|" + "|".join("---" for _ in sets) + "|"]
    for m in models:
        cells = []
        for s in sets:
            r = idx.get((m, s))
            cells.append("n/a" if r is None else f"{100 * r['wer']:.1f} [{100 * r['wer_ci_lo']:.1f}-{100 * r['wer_ci_hi']:.1f}]")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "# CER % | lenient-WER % (nukta/chandrabindu folded)\n",
              "| model | " + " | ".join(sets) + " |", "|---|" + "|".join("---" for _ in sets) + "|"]
    for m in models:
        cells = []
        for s in sets:
            r = idx.get((m, s))
            cells.append("n/a" if r is None else f"{100 * r['cer']:.1f} / {100 * r['wer_lenient']:.1f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "# Test-set sizes\n", "| set | role | utts | reference words |", "|---|---|---|---|"]
    for s in sets:
        r = next(x for x in rows if x["set"] == s)
        lines.append(f"| {s} | {role_of[s]} | {r['n_utts']} | {r['ref_words']} |")
    (res / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    eb = ["# Error breakdown (% of reference words) and speed\n",
          "| model | set | WER | sub | del | ins | empty-hyp % | RTF |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        eb.append(f"| {r['model']} | {r['set']} | {100 * r['wer']:.1f} | {100 * r['sub']:.1f} | {100 * r['del']:.1f} | "
                  f"{100 * r['ins']:.1f} | {r['empty_hyp_pct']:.1f} | {r['rtf']:.3f} |")
    train_info = []
    for m in models:
        for cand in (Path(args.runs) / m / "train_summary.json",):
            if cand.exists():
                t = json.loads(cand.read_text())
                train_info.append(f"| {m} | {t['architecture']} | {t['params'] / 1e6:.0f}M | {t['train_hours']:.1f} | "
                                  f"{t['max_steps']} | {t['wall_seconds_this_run'] / 3600:.2f} |")
    if train_info:
        eb += ["", "# Trained models\n", "| model | architecture | params | train h | steps | wall h (last run) |",
               "|---|---|---|---|---|---|"] + train_info
    (res / "errors_breakdown.md").write_text("\n".join(eb) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(sets)), 4.5))
        width = 0.8 / max(len(models), 1)
        x = np.arange(len(sets))
        for k, m in enumerate(models):
            vals = [100 * idx[(m, s)]["wer"] if (m, s) in idx else np.nan for s in sets]
            lo = [100 * (idx[(m, s)]["wer"] - idx[(m, s)]["wer_ci_lo"]) if (m, s) in idx else 0 for s in sets]
            hi = [100 * (idx[(m, s)]["wer_ci_hi"] - idx[(m, s)]["wer"]) if (m, s) in idx else 0 for s in sets]
            ax.bar(x + k * width, vals, width, yerr=[lo, hi], capsize=2, label=m)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([f"{s}\n({role_of[s]})" for s in sets], fontsize=8)
        ax.set_ylabel("WER (%)")
        ax.set_title("Hindi ASR: WER by model and test set (lower is better)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(res / "wer_chart.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("chart skipped:", e)
    print((res / "summary.md").read_text())
    print((res / "errors_breakdown.md").read_text())


if __name__ == "__main__":
    main()
