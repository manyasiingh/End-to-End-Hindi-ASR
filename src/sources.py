"""Registry + robust loaders for the Hugging Face speech datasets used here.

NOTE: Common Voice is NOT on the Hugging Face Hub any more (moved to Mozilla
Data Collective in Oct 2025). It is supported through a local directory
(see prepare_data.py --commonvoice_dir).

Every loader tries several strategies and reports exactly which one worked, so
failures are diagnosable instead of silent.
"""
from __future__ import annotations

import os

SOURCES = {
    # In-domain: read speech, news-style sentences. Gated: accept terms on the HF page.
    "kathbath": {
        "repo": "ai4bharat/Kathbath",
        "config": "hindi",
        "text_prefs": ["text"],
        "strategies": ["default"],
        "split_candidates": {
            "train": ["train"],
            "valid": ["valid", "validation"],
            "test_known": ["test_known"],
            "test_unknown": ["test_unknown"],
        },
    },
    # OOD #1: read Wikipedia-style sentences (FLORES), different speakers/mics/vocabulary.
    "fleurs": {
        "repo": "google/fleurs",
        "config": "hi_in",
        "text_prefs": ["raw_transcription", "transcription"],
        "strategies": ["default", "trust_remote_code", "parquet_branch"],
        "split_candidates": {"test": ["test"], "validation": ["validation"], "train": ["train"]},
    },
    # OOD #2: accented Hindi, read + extempore. Gated.
    "lahaja": {
        "repo": "ai4bharat/Lahaja",
        "config": None,
        "text_prefs": ["text", "transcript", "transcription", "sentence"],
        "strategies": ["default"],
        "split_candidates": {"test": ["test"]},
    },
    # OOD #3: spontaneous / conversational Hindi (hardest). Gated, large -> always streamed + capped.
    "indicvoices": {
        "repo": "ai4bharat/IndicVoices",
        "config": "hindi",
        "text_prefs": ["text", "normalized", "verbatim"],
        "strategies": ["default"],
        "split_candidates": {"valid": ["valid", "validation"], "test": ["test"]},
    },
}

GENERIC_TEXT_COLS = ["text", "transcription", "sentence", "transcript", "normalized", "verbatim"]


def available_splits(key: str) -> list[str]:
    from datasets import get_dataset_split_names
    s = SOURCES[key]
    try:
        if s["config"]:
            return list(get_dataset_split_names(s["repo"], s["config"]))
        return list(get_dataset_split_names(s["repo"]))
    except Exception:
        return []


def resolve_split(key: str, logical: str, avail: list[str] | None = None) -> str | None:
    """Map our logical split name to the real one on the Hub (None if absent)."""
    cands = SOURCES[key]["split_candidates"].get(logical, [logical])
    avail = available_splits(key) if avail is None else avail
    if not avail:  # could not list splits -> optimistic: try the first candidate
        return cands[0]
    for c in cands:
        if c in avail:
            return c
    return None


def load_split(key: str, split: str, streaming: bool = False, shuffle: bool = False,
               seed: int = 42, buffer_size: int = 2000):
    from datasets import load_dataset
    s = SOURCES[key]
    repo, cfg = s["repo"], s["config"]
    errors = []
    for strat in s["strategies"]:
        try:
            args = [repo] + ([cfg] if cfg else [])
            if strat == "default":
                ds = load_dataset(*args, split=split, streaming=streaming)
            elif strat == "trust_remote_code":
                ds = load_dataset(*args, split=split, streaming=streaming, trust_remote_code=True)
            elif strat == "parquet_branch":
                url = f"hf://datasets/{repo}@refs%2Fconvert%2Fparquet/{cfg}/{split}/*.parquet"
                ds = load_dataset("parquet", data_files={split: url}, split=split, streaming=streaming)
            else:
                raise ValueError(strat)
            print(f"[sources] {key}/{split}: loaded with strategy '{strat}' (streaming={streaming})", flush=True)
            if shuffle:
                # streaming: approximate (shard order + buffer). map-style: exact permutation, so capped
                # subsets are random draws rather than "the first N rows" (which are speaker-clustered).
                ds = ds.shuffle(seed=seed, buffer_size=buffer_size) if streaming else ds.shuffle(seed=seed)
            return ds
        except Exception as e:  # noqa: BLE001
            errors.append(f"  - {strat}: {type(e).__name__}: {str(e)[:300]}")
    hint = ""
    if any(w in " ".join(errors).lower() for w in ("gated", "401", "403", "access", "token", "authenticat")):
        hint = ("\n  HINT: this looks like an access problem. Log in (`hf auth login` or set HF_TOKEN) AND "
                f"accept the terms on https://huggingface.co/datasets/{repo}")
    raise RuntimeError(f"Could not load {key}/{split}:\n" + "\n".join(errors) + hint)


def detect_columns(example: dict, key: str | None = None, text_col: str | None = None,
                   audio_col: str | None = None) -> tuple[str, str]:
    if audio_col is None:
        for k, v in example.items():
            if isinstance(v, dict) and "array" in v and "sampling_rate" in v:
                audio_col = k
                break
    if audio_col is None:
        raise KeyError(f"No decoded audio column found. Columns: {list(example)}")
    if text_col is None:
        prefs = (SOURCES[key]["text_prefs"] if key else []) + GENERIC_TEXT_COLS
        for c in prefs:
            if c in example and isinstance(example[c], str):
                text_col = c
                break
    if text_col is None:
        raise KeyError(f"No text column found. Columns: {list(example)}")
    return audio_col, text_col


def hf_token_present() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token
        return bool(get_token())
    except Exception:  # noqa: BLE001
        return False
