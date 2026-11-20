#!/usr/bin/env python
"""Pre-flight: Python/library versions, GPU, disk, HF token. Exit code 2 if the machine cannot do the full run."""
from __future__ import annotations

import argparse
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_cpu", action="store_true", help="smoke tests only")
    ap.add_argument("--min_free_gb", type=float, default=80.0)
    ap.add_argument("--path", default=".")
    args = ap.parse_args()
    bad = []

    print(f"python {sys.version.split()[0]}")
    for pkg in ["torch", "transformers", "datasets", "accelerate", "soundfile", "librosa", "rapidfuzz",
                "jiwer", "numpy", "pandas", "huggingface_hub", "matplotlib"]:
        try:
            print(f"  {pkg:16s} {version(pkg)}")
        except PackageNotFoundError:
            print(f"  {pkg:16s} MISSING")
            bad.append(f"missing package {pkg}")
    try:
        import datasets
        import transformers
        if int(datasets.__version__.split(".")[0]) >= 4:
            bad.append("datasets>=4 detected: this pipeline is pinned to datasets<4 (script-based loaders, audio decoding)")
        if int(transformers.__version__.split(".")[0]) >= 5:
            bad.append("transformers>=5 detected: pipeline is pinned to <5 (TrainingArguments/Trainer API)")
    except Exception:  # noqa: BLE001
        pass

    import torch
    cuda = torch.cuda.is_available()
    print(f"cuda available: {cuda}")
    if cuda:
        p = torch.cuda.get_device_properties(0)
        gb = p.total_memory / 1e9
        print(f"  GPU0: {p.name}  {gb:.1f} GB  bf16={torch.cuda.is_bf16_supported()}  n_gpu={torch.cuda.device_count()}")
        if gb < 15:
            bad.append(f"GPU has {gb:.0f} GB; need >= 16 GB for full fine-tuning (lower --bs / use --accum to compensate)")
    elif not args.allow_cpu:
        bad.append("no CUDA GPU: full training is not feasible on CPU")

    free = shutil.disk_usage(args.path).free / 1e9
    print(f"free disk at {Path(args.path).resolve()}: {free:.0f} GB")
    if free < args.min_free_gb and not args.allow_cpu:
        bad.append(f"only {free:.0f} GB free; recommend >= {args.min_free_gb:.0f} GB (audio + HF cache + checkpoints)")

    from sources import hf_token_present
    print(f"HF token present: {hf_token_present()}")
    if not hf_token_present() and not args.allow_cpu:
        bad.append("no HF token: run `hf auth login` (needed for gated Kathbath / Lahaja / IndicVoices)")

    if bad:
        print("\nPROBLEMS:")
        for b in bad:
            print("  -", b)
        sys.exit(2)
    print("\nenvironment OK")


if __name__ == "__main__":
    main()
