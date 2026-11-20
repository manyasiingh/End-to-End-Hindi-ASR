"""Offline test of the data-prep logic using a fake HF dataset (no network).
Run: python tests/test_prepare.py"""
import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import SR, load_audio, read_manifest, read_meta  # noqa: E402
from evaluate import make_batches  # noqa: E402
from prepare_data import SplitWriter, ingest_hf, prep_commonvoice  # noqa: E402
from sources import detect_columns  # noqa: E402

rng = np.random.default_rng(0)


def tone(sec, sr, f=440.0):
    t = np.arange(int(sec * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # --- fake HF dataset: 22.05 kHz audio, mixed durations, one empty text, punctuation, danda
    sr0 = 22050
    durs = [2.0, 3.0, 0.2, 40.0, 1.5, 2.5, 4.0, 1.0, 2.2, 3.3]
    texts = ["नमस्ते, दुनिया।", "आज १२ बजे हैं॥", "छोटा", "बहुत लंबा", "", "ठीक है", "एक दो तीन", "चार पाँच", "छह सात", "आठ नौ"]
    ds = Dataset.from_dict({"audio_filepath": [{"array": tone(d, sr0), "sampling_rate": sr0} for d in durs],
                            "text": texts, "lang": ["hi"] * len(durs)}).cast_column("audio_filepath", Audio())

    ex = next(iter(ds))
    acol, tcol = detect_columns(ex, "kathbath")
    assert (acol, tcol) == ("audio_filepath", "text"), (acol, tcol)

    w = SplitWriter(tmp, "unit_all", "train")
    ingest_hf(ds, "kathbath", [w])
    meta = w.close()
    assert meta["skipped"] == {"short": 1, "long": 1, "empty_text": 1, "bad_audio": 0}, meta["skipped"]
    assert meta["n_utts"] == 7, meta
    recs = read_manifest(tmp / "unit_all")
    assert recs[0]["text"] == "नमस्ते दुनिया" and recs[1]["text"] == "आज 12 बजे हैं", recs[:2]
    a = load_audio(recs[0]["path"])
    info = sf.info(recs[0]["path"])
    assert info.samplerate == SR and info.channels == 1, info
    assert abs(len(a) / SR - 2.0) < 0.01 and abs(recs[0]["duration"] - 2.0) < 0.01
    assert read_meta(tmp / "unit_all")["role"] == "train"

    # --- caps + routing (the "carve val/test from valid" fallback)
    wv = SplitWriter(tmp, "unit_val", "val", max_utts=2)
    wt = SplitWriter(tmp, "unit_test", "in_domain", max_utts=3)
    ingest_hf(ds, "kathbath", [wv, wt], route=lambda i: wv if i % 5 == 0 else wt)
    wv.close(); wt.close()
    assert wv.n <= 2 and wt.n <= 3 and wv.n + wt.n >= 3, (wv.n, wt.n)

    # --- hour cap
    wh = SplitWriter(tmp, "unit_hours", "train", max_hours=5.0 / 3600)  # 5 seconds
    ingest_hf(ds, "kathbath", [wh]); wh.close()
    assert 5.0 <= wh.sec < 5.0 + 4.1 and wh.n < 7, (wh.sec, wh.n)

    # --- local Common Voice loader (wav files stand in for mp3)
    cv = tmp / "cv"; (cv / "clips").mkdir(parents=True)
    rows = []
    for i in range(4):
        sf.write(cv / "clips" / f"c{i}.wav", tone(2.0 + i * 0.1, 48000), 48000)
        rows.append({"client_id": "x", "path": f"c{i}.wav", "sentence": f"वाक्य {i}।", "up_votes": 2})
    with open(cv / "test.tsv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t"); wr.writeheader(); wr.writerows(rows)
    ns = argparse.Namespace(commonvoice_dir=str(cv), commonvoice_split="test", test_utts=10, min_dur=0.5, max_dur=30.0)
    prep_commonvoice(ns, tmp)
    r = read_manifest(tmp / "commonvoice_test")
    assert len(r) == 4 and r[0]["text"] == "वाक्य 0" and abs(r[0]["duration"] - 2.0) < 0.01, r[:1]
    assert read_meta(tmp / "commonvoice_test")["role"] == "ood"

# --- batching: ascending durations; any multi-item batch must satisfy len * max_dur <= max_sec
d = [1, 1, 2, 5, 10, 10, 30]
b = make_batches(d, max_bs=3, max_sec=20)
assert sorted(i for x in b for i in x) == list(range(7)), b
assert all(len(x) <= 3 for x in b), b
assert all(len(x) == 1 or len(x) * d[x[-1]] <= 20 for x in b), b
assert [6] in b  # the 30 s item cannot share a batch
print("prepare/data tests OK")
