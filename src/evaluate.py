#!/usr/bin/env python
"""Run models over test sets and store raw predictions (metrics are computed in report.py).

--model NAME=KIND:PATH   (repeatable)
   KIND = whisper | ctc | indicconformer
   whisper : HF Whisper dir or hub id (zero-shot or fine-tuned)
   ctc     : dir written by train_ctc.py (model + vocab + feature extractor)
   indicconformer : ai4bharat/indic-conformer-600m-multilingual (gated, trust_remote_code, zero-shot reference)

Output: results/<NAME>/<set>.jsonl  (id, ref, hyp) and results/<NAME>/<set>.timing.json
Existing outputs are skipped unless --force (so long jobs can be resumed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SR, load_audio, read_manifest, read_meta  # noqa: E402
from textnorm import normalize  # noqa: E402


def make_batches(durations, max_bs, max_sec):
    """durations already sorted ascending. Greedy batches bounded by count and padded seconds."""
    batches, cur = [], []
    for i, d in enumerate(durations):
        if cur and (len(cur) >= max_bs or (len(cur) + 1) * d > max_sec):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


class WhisperASR:
    def __init__(self, path, device, num_beams=1, max_new_tokens=400):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.proc = WhisperProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(path).to(device=device, dtype=self.dtype).eval()
        self.model.config.use_cache = True
        self.num_beams, self.max_new_tokens = num_beams, max_new_tokens
        self.max_bs, self.max_sec = 16, float("inf")  # Whisper pads every item to 30 s anyway

    @torch.inference_mode()
    def transcribe(self, audios):
        feats = self.proc.feature_extractor(audios, sampling_rate=SR, return_tensors="pt").input_features
        ids = self.model.generate(feats.to(self.device, self.dtype), language="hi", task="transcribe",
                                  num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
        return self.proc.batch_decode(ids, skip_special_tokens=True)


class CTCASR:
    def __init__(self, path, device):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.proc = Wav2Vec2Processor.from_pretrained(path)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).to(device=device, dtype=self.dtype).eval()
        self.use_mask = self.model.config.feat_extract_norm == "layer"
        self.max_bs, self.max_sec = 32, 400.0

    @torch.inference_mode()
    def transcribe(self, audios):
        b = self.proc.feature_extractor(audios, sampling_rate=SR, padding=True,
                                        return_attention_mask=self.use_mask, return_tensors="pt")
        kw = {"attention_mask": b["attention_mask"].to(self.device)} if self.use_mask else {}
        logits = self.model(b["input_values"].to(self.device, self.dtype), **kw).logits
        ids = torch.argmax(logits, dim=-1).cpu().numpy()
        # Decode only each utterance's own frames: the frames that correspond to batch zero-padding produce
        # junk symbols (seen as stray characters appended to the hypothesis) even with an attention mask.
        out_len = self.model._get_feat_extract_output_lengths(torch.tensor([len(a) for a in audios])).tolist()
        return [self.proc.tokenizer.decode(ids[i, :min(int(n), ids.shape[1])]) for i, n in enumerate(out_len)]


class IndicConformerASR:
    """Zero-shot reference. Uses the repo's remote code: model(wav[1,T], 'hi', 'ctc')."""

    def __init__(self, path, device, decoder="ctc"):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True)
        self.decoder = decoder
        self.max_bs, self.max_sec = 1, float("inf")
        self.failures = 0

    def transcribe(self, audios):
        outs = []
        for a in audios:
            try:
                r = self.model(torch.from_numpy(a).unsqueeze(0), "hi", self.decoder)
                if isinstance(r, (list, tuple)):
                    r = r[0]
                outs.append(str(r))
            except Exception as e:  # noqa: BLE001
                self.failures += 1
                if self.failures <= 3:
                    print("[indicconformer] utterance failed:", type(e).__name__, str(e)[:200])
                outs.append("")
        return outs


def build(kind, path, device, args):
    if kind == "whisper":
        return WhisperASR(path, device, num_beams=args.num_beams)
    if kind == "ctc":
        return CTCASR(path, device)
    if kind == "indicconformer":
        return IndicConformerASR(path, device)
    raise ValueError(f"unknown model kind {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=KIND:PATH")
    ap.add_argument("--data", default="data")
    ap.add_argument("--sets", nargs="*", default=None, help="set dir names; default = every test/OOD set found")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max_utts", type=int, default=None, help="cap per set (debug / slow models)")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    if args.sets:
        sets = args.sets
    else:
        sets = sorted(p.name for p in data.iterdir()
                      if (p / "meta.json").exists() and read_meta(p).get("role") in ("in_domain", "ood"))
    print(f"[evaluate] device={device} sets={sets}", flush=True)
    if not sets:
        raise SystemExit("no evaluation sets found - run prepare_data.py")

    failed = []
    for spec in args.model:
        name, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        if not Path(path).exists() and (path.startswith(("/", "./", "../", "runs")) or path.count("/") == 0):
            failed.append(f"{name}: local model dir '{path}' does not exist (did training finish?)")
            print(f"[evaluate] {failed[-1]}", flush=True)
            continue
        todo = [s for s in sets if args.force or not (Path(args.out) / name / f"{s}.jsonl").exists()]
        if not todo:
            print(f"[evaluate] {name}: all sets done, skipping", flush=True)
            continue
        print(f"\n[evaluate] loading {name} ({kind}) from {path}", flush=True)
        try:
            asr = build(kind, path, device, args)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            failed.append(f"{name}: could not load ({type(e).__name__}: {str(e)[:200]})")
            continue
        for s in todo:
            recs = read_manifest(data / s, max_utts=args.max_utts)
            recs.sort(key=lambda r: r["duration"])
            batches = make_batches([r["duration"] for r in recs], asr.max_bs, asr.max_sec)
            hyps = [None] * len(recs)
            audio_sec = sum(r["duration"] for r in recs)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            try:
                for b in tqdm(batches, desc=f"{name} | {s}", leave=False):
                    audios = [load_audio(recs[i]["path"]) for i in b]
                    for i, h in zip(b, asr.transcribe(audios)):
                        hyps[i] = h
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                failed.append(f"{name}/{s}: {type(e).__name__}: {str(e)[:200]}")
                continue
            if device == "cuda":
                torch.cuda.synchronize()
            wall = time.time() - t0
            d = Path(args.out) / name
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{s}.jsonl", "w", encoding="utf-8") as f:
                for r, h in zip(recs, hyps):
                    f.write(json.dumps({"id": r["id"], "duration": r["duration"], "ref": r["text"],
                                        "hyp_raw": h, "hyp": normalize(h)}, ensure_ascii=False) + "\n")
            (d / f"{s}.timing.json").write_text(json.dumps(
                {"audio_seconds": audio_sec, "wall_seconds": wall, "rtf": wall / max(audio_sec, 1e-9),
                 "device": device, "n_utts": len(recs)}, indent=1), encoding="utf-8")
            print(f"[evaluate] {name} | {s}: {len(recs)} utts, RTF {wall / max(audio_sec, 1e-9):.3f}", flush=True)
        del asr
        if device == "cuda":
            torch.cuda.empty_cache()

    if failed:
        print("\n[evaluate] FAILURES:")
        for f in failed:
            print("  -", f)
        sys.exit(3)


if __name__ == "__main__":
    main()
