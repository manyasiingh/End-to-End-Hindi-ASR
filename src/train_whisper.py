#!/usr/bin/env python
"""Architecture A: Whisper (encoder-decoder transformer, autoregressive, cross-entropy loss).
Full fine-tuning (all weights) on the shared Kathbath-Hindi manifest.

Targets are the SAME normalised text the CTC model sees (no punctuation), so the
comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor, set_seed)
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SR, count_params, load_audio, read_manifest  # noqa: E402


@dataclass
class WhisperCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, feats):
        audios = [load_audio(f["path"]) for f in feats]
        inputs = self.processor.feature_extractor(audios, sampling_rate=SR, return_tensors="pt")
        label_ids = [self.processor.tokenizer(f["text"]).input_ids for f in feats]
        padded = self.processor.tokenizer.pad({"input_ids": label_ids}, return_tensors="pt")
        labels = padded["input_ids"].masked_fill(padded["attention_mask"].ne(1), -100)
        # The model prepends decoder_start_token_id itself when shifting labels right.
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": inputs["input_features"], "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/kathbath_train")
    ap.add_argument("--val_dir", default="data/kathbath_val")
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--out", default="runs/whisper_small")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--bs", type=int, default=16, help="per-device train batch size")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--max_val", type=int, default=800)
    ap.add_argument("--min_dur", type=float, default=0.5)
    ap.add_argument("--max_dur", type=float, default=30.0)
    ap.add_argument("--max_label_tokens", type=int, default=440, help="Whisper decoder limit is 448")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    use_bf16 = cuda and torch.cuda.is_bf16_supported()
    use_fp16 = cuda and not use_bf16
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[train_whisper] cuda={cuda} bf16={use_bf16} fp16={use_fp16}", flush=True)

    processor = WhisperProcessor.from_pretrained(args.model, language="hindi", task="transcribe")
    tok = processor.tokenizer
    if tok.pad_token is None:  # real Whisper tokenizers define it; guard for custom ones
        tok.pad_token = tok.eos_token
    tok.set_prefix_tokens(language="hindi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "hindi"
    model.generation_config.task = "transcribe"
    model.config.use_cache = False  # incompatible with gradient checkpointing; re-enabled at inference

    train = read_manifest(args.train_dir, min_dur=args.min_dur, max_dur=args.max_dur)
    val = read_manifest(args.val_dir, max_utts=args.max_val, min_dur=args.min_dur, max_dur=args.max_dur)
    # Drop examples whose transcript would not fit the decoder.
    def keep_fit(recs, name):
        lens = [len(x) for x in tok([r["text"] for r in recs]).input_ids]
        kept = [r for r, n in zip(recs, lens) if n <= args.max_label_tokens]
        print(f"[train_whisper] {name}: kept {len(kept)}/{len(recs)} "
              f"(dropped {len(recs) - len(kept)} with > {args.max_label_tokens} label tokens)", flush=True)
        return kept
    train, val = keep_fit(train, "train"), keep_fit(val, "val")
    if not train or not val:
        raise SystemExit("empty train/val set - run prepare_data.py first")
    print(f"[train_whisper] train {len(train)} utts / {sum(r['duration'] for r in train) / 3600:.1f} h; "
          f"val {len(val)} utts", flush=True)

    keys = ("path", "text", "duration")
    train_ds = Dataset.from_list([{k: r[k] for k in keys} for r in train])
    val_ds = Dataset.from_list([{k: r[k] for k in keys} for r in val])

    start_id = model.config.decoder_start_token_id
    collator = WhisperCollator(processor, start_id)
    # Sanity print: first training target as the decoder will see it.
    ex = collator([train_ds[0]])
    print("[train_whisper] sample target:", repr(tok.decode(ex["labels"][0].masked_fill(ex["labels"][0] < 0, tok.pad_token_id),
                                                           skip_special_tokens=False))[:200], flush=True)

    args_tr = Seq2SeqTrainingArguments(
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
        bf16=use_bf16,
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.workers,
        predict_with_generate=False,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(model=model, args=args_tr, train_dataset=train_ds, eval_dataset=val_ds,
                             data_collator=collator)

    last = None if args.no_resume else get_last_checkpoint(str(out))
    if last:
        print(f"[train_whisper] resuming from {last}", flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last)
    wall = time.time() - t0

    trainer.save_model(str(out))       # best checkpoint (load_best_model_at_end)
    processor.save_pretrained(str(out))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    summary = {
        "architecture": "whisper (encoder-decoder, attention, CE loss)",
        "base_model": args.model, "params": count_params(model),
        "train_hours": sum(r["duration"] for r in train) / 3600, "train_utts": len(train),
        "max_steps": args.max_steps, "effective_batch": args.bs * args.accum, "lr": args.lr,
        "wall_seconds_this_run": wall, "best_metric": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[train_whisper] done:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
