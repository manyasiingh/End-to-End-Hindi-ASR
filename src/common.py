"""Small shared helpers: manifests, audio loading."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def read_meta(set_dir) -> dict:
    p = Path(set_dir) / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_manifest(set_dir, max_utts: int | None = None, min_dur: float = 0.0,
                  max_dur: float = 1e9) -> list[dict]:
    """Each record: id, audio (relative), text_raw, text (normalised), duration, path (absolute)."""
    set_dir = Path(set_dir)
    recs = []
    with open(set_dir / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not (min_dur <= r["duration"] <= max_dur):
                continue
            r["path"] = str((set_dir / r["audio"]).resolve())
            recs.append(r)
            if max_utts and len(recs) >= max_utts:
                break
    return recs


def resample_to_16k(arr, sr: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:  # (T, C) from soundfile or (C, T) from some loaders -> mono
        arr = arr.mean(axis=1 if arr.shape[0] > arr.shape[-1] else 0)
    if int(sr) != SR:
        import librosa
        arr = librosa.resample(arr, orig_sr=int(sr), target_sr=SR)
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_audio(path) -> np.ndarray:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return resample_to_16k(arr, sr)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
