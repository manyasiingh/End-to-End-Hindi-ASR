#!/usr/bin/env python
"""Architecture B: wav2vec2-style encoder + CTC head (non-autoregressive, frame-level, CTC loss).
Default init: facebook/mms-300m (self-supervised on 1,400+ languages, includes Hindi).
Character vocabulary is built from the TRAIN transcripts (Devanagari code points incl. matras).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (Trainer, TrainingArguments, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2ForCTC, Wav2Vec2Processor, set_seed)
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SR, count_params, load_audio, read_manifest  # noqa: E402
from metrics import wer_cer  # noqa: E402


def build_vocab(texts: list[str], out_dir: Path) -> Path:
    chars = sorted({c for t in texts for c in t if c != " "})
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["|"] = len(vocab)        # word delimiter (space)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)    # also the CTC blank
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "vocab.json"
    p.write_text(json.dumps(vocab, ensure_ascii=False, indent=0), encoding="utf-8")
    return p


@dataclass
class CTCCollator:
    processor: Wav2Vec2Processor
    use_attention_mask: bool

    def __call__(self, feats):
        audios = [load_audio(f["path"]) for f in feats]
        batch = self.processor.feature_extractor(
            audios, sampling_rate=SR, padding=True, return_attention_mask=self.use_attention_mask,
            return_tensors="pt")
        lab = self.processor.tokenizer([f["text"] for f in feats]).input_ids
        padded = self.processor.tokenizer.pad({"input_ids": lab}, padding=True, return_tensors="pt")
        labels = padded["input_ids"].masked_fill(padded["attention_mask"].ne(1), -100)
        out = {"input_values": batch["input_values"], "labels": labels}
        if self.use_attention_mask:
            out["attention_mask"] = batch["attention_mask"]
        return out


class CTCTrainer(Trainer):
    """Validation only: decode each utterance over its own frames. Without this, the frames that belong to the
    batch zero-padding of shorter clips are argmax-decoded too and append junk characters to the hypothesis,
    inflating the logged validation WER (and biasing checkpoint selection). Training/loss are untouched."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if logits is None:
            return loss, logits, labels
        if isinstance(logits, tuple):
            logits = logits[0]
        ids = torch.argmax(logits, dim=-1)
        mask = inputs.get("attention_mask")
        if mask is not None:  # layer-norm models (MMS, XLS-R) always get the mask; true sample lengths
            base = self.accelerator.unwrap_model(model)
            out_len = base._get_feat_extract_output_lengths(mask.sum(-1)).to(ids.device)
            pad_frames = torch.arange(ids.shape[1], device=ids.device)[None, :] >= out_len[:, None]
            ids = ids.masked_fill(pad_frames, self.processing_class_pad_id)
        return loss, ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/kathbath_train")
    ap.add_argument("--val_dir", default="data/kathbath_val")
    ap.add_argument("--model", default="facebook/mms-300m")
    ap.add_argument("--out", default="runs/mms300m_ctc")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=12000)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--max_val", type=int, default=500)
    ap.add_argument("--min_dur", type=float, default=0.5)
    ap.add_argument("--max_dur", type=float, default=25.0, help="memory guard for training only")
    ap.add_argument("--mask_time_prob", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--no_group_by_length", action="store_true")
    ap.add_argument("--train_feature_encoder", action="store_true",
                    help="unfreeze the CNN front-end (default: frozen, the standard recipe)")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    use_bf16 = cuda and torch.cuda.is_bf16_supported()
    use_fp16 = cuda and not use_bf16
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[train_ctc] cuda={cuda} bf16={use_bf16} fp16={use_fp16}", flush=True)

    train = read_manifest(args.train_dir, min_dur=args.min_dur, max_dur=args.max_dur)
    val = read_manifest(args.val_dir, max_utts=args.max_val, min_dur=args.min_dur, max_dur=args.max_dur)
    if not train or not val:
        raise SystemExit("empty train/val set - run prepare_data.py first")
    print(f"[train_ctc] train {len(train)} utts / {sum(r['duration'] for r in train) / 3600:.1f} h; val {len(val)}",
          flush=True)

    vocab_path = build_vocab([r["text"] for r in train], out)
    tokenizer = Wav2Vec2CTCTokenizer(str(vocab_path), unk_token="[UNK]", pad_token="[PAD]",
                                     word_delimiter_token="|")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    processor = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tokenizer)
    print(f"[train_ctc] vocab size {len(tokenizer)}: {list(tokenizer.get_vocab())[:12]} ...", flush=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, final_dropout=0.0,
        layerdrop=0.0,
        mask_time_prob=args.mask_time_prob, mask_time_length=10,
    )
    if not args.train_feature_encoder:
        model.freeze_feature_encoder()  # standard: keep the CNN front-end frozen
    use_mask = model.config.feat_extract_norm == "layer"  # layer-norm models need the attention mask
    collator = CTCCollator(processor, use_attention_mask=use_mask)

    keys = ("path", "text", "duration")
    train_ds = Dataset.from_list([{k: r[k] for k in keys} for r in train])
    val_ds = Dataset.from_list([{k: r[k] for k in keys} for r in val])
    ex = collator([train_ds[0]])
    print("[train_ctc] sample label ids:", ex["labels"][0].tolist()[:30], flush=True)
    print("[train_ctc] decoded back   :", repr(tokenizer.decode(ex["labels"][0].tolist(), group_tokens=False))[:120],
          "| original:", repr(train[0]["text"])[:120], flush=True)

    pad_id = tokenizer.pad_token_id

    def compute_metrics(pred):
        ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        ids = np.where(ids == -100, pad_id, ids)
        lab = np.where(pred.label_ids == -100, pad_id, pred.label_ids)
        hyp = tokenizer.batch_decode(ids)
        ref = tokenizer.batch_decode(lab, group_tokens=False)
        wer, cer = wer_cer(ref, hyp)
        return {"wer": wer, "cer": cer}

    # (argmax + padded-frame masking now happens in CTCTrainer.prediction_step)

    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lr_scheduler_type="linear",
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=not args.no_group_by_length,
        length_column_name="duration",
        bf16=use_bf16,
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.workers,
        seed=args.seed,
    )
    trainer = CTCTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                         data_collator=collator, compute_metrics=compute_metrics)
    trainer.processing_class_pad_id = pad_id  # CTC blank; padded frames decode to nothing

    last = None if args.no_resume else get_last_checkpoint(str(out))
    if last:
        print(f"[train_ctc] resuming from {last}", flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last)
    wall = time.time() - t0

    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    summary = {
        "architecture": "wav2vec2 encoder + CTC head (non-autoregressive)",
        "base_model": args.model, "params": count_params(model),
        "train_hours": sum(r["duration"] for r in train) / 3600, "train_utts": len(train),
        "max_steps": args.max_steps, "effective_batch": args.bs * args.accum, "lr": args.lr,
        "wall_seconds_this_run": wall, "best_metric": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint, "vocab_size": len(tokenizer),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[train_ctc] done:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
