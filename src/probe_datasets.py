#!/usr/bin/env python
"""Cheap pre-flight: can we reach/auth each dataset, what splits/columns exist, does audio decode?
Reads ONE example per split (streaming) - no big downloads. Run this before prepare_data.py."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import SOURCES, available_splits, detect_columns, hf_token_present, load_split  # noqa: E402
from textnorm import normalize  # noqa: E402


def main():
    print(f"HF token present: {hf_token_present()}")
    if hf_token_present():
        try:
            from huggingface_hub import HfApi
            print("Logged in as:", HfApi().whoami().get("name"))
        except Exception as e:  # noqa: BLE001
            print("whoami failed:", type(e).__name__, str(e)[:200])
    else:
        print("!! No HF token. Gated datasets (Kathbath, Lahaja, IndicVoices) will fail. Run `hf auth login`.")

    status = {}
    for key, s in SOURCES.items():
        print(f"\n--- {key}  ({s['repo']}, config={s['config']}) ---")
        avail = available_splits(key)
        print("splits:", avail or "(could not list)")
        tried = [sp for sp in (avail or [c[0] for c in s["split_candidates"].values()])][:3]
        ok = False
        for sp in tried:
            try:
                ds = load_split(key, sp, streaming=True)
                ex = next(iter(ds))
                a, t = detect_columns(ex, key)
                arr, sr = ex[a]["array"], ex[a]["sampling_rate"]
                print(f"[{sp}] columns={list(ex.keys())}")
                print(f"[{sp}] audio_col={a} sr={sr} dur={len(arr) / sr:.2f}s  text_col={t}")
                print(f"[{sp}] text_raw ={ex[t]!r}")
                print(f"[{sp}] text_norm={normalize(ex[t])!r}")
                ok = True
            except Exception as e:  # noqa: BLE001
                print(f"[{sp}] FAILED: {type(e).__name__}: {str(e)[:400]}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
        status[key] = ok
    print("\n=========== PROBE SUMMARY ===========")
    for k, v in status.items():
        print(f"{k:12s} {'OK' if v else 'NOT USABLE'}")
    if not status.get("kathbath"):
        print("\nKathbath (training data) is not usable -> fix access before continuing.")
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
