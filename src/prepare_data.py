#!/usr/bin/env python
"""Build local, model-agnostic datasets (16 kHz mono FLAC + manifest.jsonl + meta.json).

Output layout (--out data):
  kathbath_train/          in-domain TRAIN (streamed + shuffled, capped by --train_hours)
  kathbath_val/            validation used during training (checkpoint selection)
  kathbath_test_known/ ... in-domain TEST (whatever test splits exist on the Hub), or
  kathbath_heldout/        fallback in-domain test carved from `valid` if no test split exists
  fleurs_test/             OOD: read Wikipedia-style Hindi
  lahaja_test/             OOD: accented Hindi (gated)
  indicvoices_valid/       OOD: spontaneous/conversational Hindi (gated; capped)
  commonvoice_test/        OOD: only if --commonvoice_dir is given (manual download)
Each set dir has meta.json with role = train | val | in_domain | ood.

Failures of OOD sources are reported but non-fatal; the Kathbath TRAIN set and at least
one test set are required for the rest of the pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SR, resample_to_16k  # noqa: E402
from sources import SOURCES, available_splits, detect_columns, load_split, resolve_split  # noqa: E402
from textnorm import normalize  # noqa: E402


class SplitWriter:
    def __init__(self, root, name, role, max_hours=None, max_utts=None,
                 min_dur=0.5, max_dur=30.0, info=None):
        self.dir = Path(root) / name
        (self.dir / "audio").mkdir(parents=True, exist_ok=True)
        self.name, self.role = name, role
        self.max_sec = max_hours * 3600 if max_hours else None
        self.max_utts = max_utts
        self.min_dur, self.max_dur = min_dur, max_dur
        self.info = info or {}
        self.n, self.sec = 0, 0.0
        self.skipped = {"short": 0, "long": 0, "empty_text": 0, "bad_audio": 0}
        self._f = open(self.dir / "manifest.jsonl", "w", encoding="utf-8")

    @property
    def full(self) -> bool:
        return ((self.max_utts is not None and self.n >= self.max_utts)
                or (self.max_sec is not None and self.sec >= self.max_sec))

    def add(self, audio, sr, text, uid=None) -> None:
        if self.full:
            return
        text_norm = normalize(text)
        if not text_norm:
            self.skipped["empty_text"] += 1
            return
        try:
            wav = resample_to_16k(audio, sr)
        except Exception:  # noqa: BLE001
            self.skipped["bad_audio"] += 1
            return
        if wav.size == 0 or not np.isfinite(wav).all():
            self.skipped["bad_audio"] += 1
            return
        dur = wav.shape[0] / SR
        if dur < self.min_dur:
            self.skipped["short"] += 1
            return
        if dur > self.max_dur:
            self.skipped["long"] += 1
            return
        rel = f"audio/{self.n:07d}.flac"
        sf.write(self.dir / rel, np.clip(wav, -1.0, 1.0), SR, format="FLAC", subtype="PCM_16")
        rec = {"id": uid or f"{self.name}_{self.n:07d}", "audio": rel, "text_raw": text,
               "text": text_norm, "duration": round(dur, 3)}
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n += 1
        self.sec += dur
        if self.n % 1000 == 0:
            print(f"    [{self.name}] {self.n} utts, {self.sec / 3600:.2f} h", flush=True)

    def close(self) -> dict:
        self._f.close()
        meta = {"name": self.name, "role": self.role, "n_utts": self.n,
                "hours": round(self.sec / 3600, 3), "skipped": self.skipped,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), **self.info}
        (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {self.name}: {self.n} utts / {self.sec / 3600:.2f} h  (skipped {self.skipped})", flush=True)
        return meta


def ingest_hf(ds, key, writers, route=None, text_col=None, audio_col=None, max_scan=None):
    """Iterate a HF dataset, send examples to writer(s). route(i)->writer; default: writers[0]."""
    acol = tcol = None
    for i, ex in enumerate(ds):
        if acol is None:
            acol, tcol = detect_columns(ex, key, text_col, audio_col)
            print(f"    columns: audio='{acol}' text='{tcol}'", flush=True)
        w = route(i) if route else writers[0]
        a = ex[acol]
        w.add(a["array"], a["sampling_rate"], ex[tcol])
        if all(x.full for x in writers):
            break
        if max_scan and i >= max_scan:
            break


def run_step(name, fn, required, results):
    print(f"\n=== {name} ===", flush=True)
    try:
        fn()
        results[name] = "ok"
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        results[name] = f"FAILED: {type(e).__name__}: {str(e)[:500]}"
        if required:
            print(f"!! REQUIRED step '{name}' failed.", flush=True)


# ----------------------------------------------------------------------------- sources
def prep_kathbath(args, out):
    key = "kathbath"
    avail = available_splits(key)
    print(f"  available splits: {avail or '(could not list; will try defaults)'}")

    # TRAIN (streamed, shuffled so we sample many speakers, not the first shards)
    sp = resolve_split(key, "train", avail)
    if sp is None:
        raise RuntimeError("Kathbath has no train split?")
    w = SplitWriter(out, "kathbath_train", "train", max_hours=args.train_hours,
                    min_dur=args.min_dur, max_dur=args.max_dur, info={"source": "kathbath", "split": sp})
    ds = load_split(key, sp, streaming=True, shuffle=True, seed=args.seed)
    ingest_hf(ds, key, [w], text_col=args.kathbath_text_col)
    w.close()

    # VAL + in-domain TEST
    # Only trust test splits that are explicitly listed on the Hub (no optimistic guessing here).
    test_splits = [(lg, resolve_split(key, lg, avail)) for lg in ("test_known", "test_unknown")] if avail else []
    test_splits = [(lg, sp) for lg, sp in test_splits if sp]
    vsp = resolve_split(key, "valid", avail)
    if test_splits:
        if vsp:
            wv = SplitWriter(out, "kathbath_val", "val", max_utts=args.val_utts, min_dur=args.min_dur,
                             max_dur=args.max_dur, info={"source": "kathbath", "split": vsp})
            ingest_hf(load_split(key, vsp, streaming=False, shuffle=True, seed=args.seed), key, [wv],
                      text_col=args.kathbath_text_col)
            wv.close()
        for lg, sp in test_splits:
            wt = SplitWriter(out, f"kathbath_{lg}", "in_domain", max_utts=args.test_utts, min_dur=args.min_dur,
                             max_dur=args.max_dur, info={"source": "kathbath", "split": sp})
            ingest_hf(load_split(key, sp, streaming=False, shuffle=True, seed=args.seed), key, [wt],
                          text_col=args.kathbath_text_col)
            wt.close()
    else:
        if not vsp:
            raise RuntimeError("Kathbath has neither test splits nor a valid split: cannot build val/test.")
        print("  No test split on the Hub -> carving val/held-out test from `valid` "
              "(every 5th utterance -> val, rest -> test). NOT speaker-disjoint; stated in meta.json.")
        wv = SplitWriter(out, "kathbath_val", "val", max_utts=args.val_utts, min_dur=args.min_dur,
                         max_dur=args.max_dur, info={"source": "kathbath", "split": vsp})
        wt = SplitWriter(out, "kathbath_heldout", "in_domain", max_utts=args.test_utts, min_dur=args.min_dur,
                         max_dur=args.max_dur,
                         info={"source": "kathbath", "split": vsp,
                               "caveat": "carved from valid split; not guaranteed speaker-disjoint from val"})
        ingest_hf(load_split(key, vsp, streaming=False, shuffle=True, seed=args.seed), key, [wv, wt],
                  route=lambda i: wv if i % 5 == 0 else wt, text_col=args.kathbath_text_col)
        wv.close()
        wt.close()


def _prep_simple_ood(key, logical_split, set_name, args, out, streaming, max_utts, max_hours=None):
    sp = resolve_split(key, logical_split)
    if sp is None:
        raise RuntimeError(f"{key}: split '{logical_split}' not found")
    w = SplitWriter(out, set_name, "ood", max_utts=max_utts, max_hours=max_hours, min_dur=args.min_dur,
                    max_dur=args.max_dur, info={"source": key, "split": sp})
    ds = load_split(key, sp, streaming=streaming, shuffle=True, seed=args.seed)
    ingest_hf(ds, key, [w])
    w.close()


def prep_commonvoice(args, out):
    """Local Common Voice (download from Mozilla Data Collective). Expects <dir>/<split>.tsv + <dir>/clips/."""
    import librosa
    d = Path(args.commonvoice_dir)
    tsv = d / f"{args.commonvoice_split}.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"{tsv} not found (expected cv-corpus-*/hi/{args.commonvoice_split}.tsv)")
    w = SplitWriter(out, "commonvoice_test", "ood", max_utts=args.test_utts, min_dur=args.min_dur,
                    max_dur=args.max_dur, info={"source": "commonvoice_local", "split": args.commonvoice_split})
    with open(tsv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        rows = list(reader)
    for r in rows:
        if w.full:
            break
        clip = d / "clips" / r["path"]
        try:
            y, sr = librosa.load(str(clip), sr=None, mono=True)
        except Exception:  # noqa: BLE001
            w.skipped["bad_audio"] += 1
            continue
        w.add(y, sr, r["sentence"])
    w.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: kathbath fleurs lahaja indicvoices commonvoice (default: all available)")
    ap.add_argument("--train_hours", type=float, default=100.0)
    ap.add_argument("--val_utts", type=int, default=1000)
    ap.add_argument("--test_utts", type=int, default=2000, help="cap per test set")
    ap.add_argument("--indicvoices_utts", type=int, default=1500)
    ap.add_argument("--min_dur", type=float, default=0.5)
    ap.add_argument("--max_dur", type=float, default=30.0, help="Whisper's window is 30 s")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kathbath_text_col", default=None)
    ap.add_argument("--commonvoice_dir", default=None, help="local cv-corpus-*/hi directory (optional)")
    ap.add_argument("--commonvoice_split", default="test")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only) if args.only else {"kathbath", "fleurs", "lahaja", "indicvoices"} | (
        {"commonvoice"} if args.commonvoice_dir else set())
    results: dict[str, str] = {}

    if "kathbath" in want:
        run_step("kathbath (train/val/in-domain test)", lambda: prep_kathbath(args, out), True, results)
    if "fleurs" in want:
        run_step("fleurs hi_in test (OOD)", lambda: _prep_simple_ood(
            "fleurs", "test", "fleurs_test", args, out, streaming=False, max_utts=args.test_utts), False, results)
    if "lahaja" in want:
        run_step("lahaja test (OOD, gated)", lambda: _prep_simple_ood(
            "lahaja", "test", "lahaja_test", args, out, streaming=False, max_utts=args.test_utts), False, results)
    if "indicvoices" in want:
        run_step("indicvoices valid (OOD, gated, streamed)", lambda: _prep_simple_ood(
            "indicvoices", "valid", "indicvoices_valid", args, out, streaming=True,
            max_utts=args.indicvoices_utts), False, results)
    if "commonvoice" in want:
        if not args.commonvoice_dir:
            results["commonvoice"] = "SKIPPED: pass --commonvoice_dir"
        else:
            run_step("commonvoice local test (OOD)", lambda: prep_commonvoice(args, out), False, results)

    print("\n================ SUMMARY ================")
    for k, v in results.items():
        print(f"{k:50s} {v}")
    sets = sorted(p.name for p in out.iterdir() if (p / "meta.json").exists())
    print("Sets on disk:", sets)
    (out / "prepare_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    kath_ok = results.get("kathbath (train/val/in-domain test)") == "ok"
    if "kathbath" in want and not kath_ok:
        sys.exit(2)


if __name__ == "__main__":
    # Streaming `datasets` leaves background threads that can abort the interpreter during shutdown
    # (Fatal Python error: PyGILState_Release, exit 134) AFTER all work is done. Exit hard instead.
    import os
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
