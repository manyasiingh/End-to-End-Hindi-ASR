"""Synthetic, *learnable* speech-like data for offline pipeline tests.
Each Devanagari letter = a tone burst of a fixed frequency; space = longer silence.
Writes the exact same layout as prepare_data.py (via SplitWriter)."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_data import SplitWriter  # noqa: E402

LETTERS = list("कखगघचजतदनपमरलवसह")  # 16 letters
SR = 16000


def synth(text, rng, noise):
    parts = [np.zeros(int(0.05 * SR))]
    for ch in text:
        if ch == " ":
            parts.append(np.zeros(int(0.12 * SR)))
            continue
        f = 250 + 110 * LETTERS.index(ch)
        t = np.arange(int(0.10 * SR)) / SR
        parts.append(0.5 * np.sin(2 * np.pi * f * t) * np.hanning(len(t)))
        parts.append(np.zeros(int(0.03 * SR)))
    y = np.concatenate(parts + [np.zeros(int(0.05 * SR))])
    return (y + noise * rng.standard_normal(len(y))).astype(np.float32)


def rand_text(rng, nwords, wlen):
    return " ".join("".join(rng.choice(LETTERS, size=rng.integers(*wlen))) for _ in range(nwords))


def make(root, name, role, n, seed, noise, nwords=(2, 3), wlen=(2, 4), max_dur=2.9):
    rng = np.random.default_rng(seed)
    w = SplitWriter(root, name, role, min_dur=0.5, max_dur=max_dur)
    while w.n < n:
        txt = rand_text(rng, rng.integers(*nwords) , wlen)
        w.add(synth(txt, rng, noise), SR, txt)
    w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/smoke/data")
    ap.add_argument("--n_train", type=int, default=600)
    a = ap.parse_args()
    make(a.out, "kathbath_train", "train", a.n_train, 1, 0.01)
    make(a.out, "kathbath_val", "val", 40, 2, 0.01)
    make(a.out, "kathbath_heldout", "in_domain", 60, 3, 0.01)
    # noisier -> out of domain. OOD_NOISE env var (default 0.08) exists only for diagnosing the smoke gate.
    import os
    make(a.out, "fleurs_test", "ood", 60, 4, float(os.environ.get("OOD_NOISE", "0.08")), wlen=(2, 4))
