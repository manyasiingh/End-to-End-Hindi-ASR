#!/usr/bin/env bash
# Offline end-to-end pipeline test: synthetic tone "speech" + tiny random models. No internet, no HF token.
# Takes a few minutes on CPU. If this passes, the code + library versions on THIS machine are sound.
set -euo pipefail
cd "$(dirname "$0")/.."
S=${SMOKE_DIR:-/tmp/indic_asr_smoke}
rm -rf "$S"; mkdir -p "$S"
export TOKENIZERS_PARALLELISM=false

echo "== unit tests"; python src/textnorm.py; python tests/test_metrics.py
echo "== synthetic data + tiny models"
python tests/make_synthetic.py --out "$S/data" --n_train 600
python tests/make_tiny_models.py "$S/models" 2>&1 | grep -vi warn

echo "== train CTC (tiny)"
python src/train_ctc.py --train_dir "$S/data/kathbath_train" --val_dir "$S/data/kathbath_val" \
  --model "$S/models/tiny_w2v2" --out "$S/runs/tiny_ctc" --lr 2e-3 --max_steps 2500 --warmup_steps 100 \
  --bs 16 --accum 1 --eval_bs 16 --eval_steps 500 --logging_steps 250 --workers 0 --mask_time_prob 0.0 \
  --train_feature_encoder --no_grad_ckpt > "$S/ctc.log" 2>&1 || { tail -40 "$S/ctc.log"; exit 1; }
grep -E "eval_wer" "$S/ctc.log" | tr '\r' '\n' | grep -oE "\{'eval_loss[^}]*\}" | tail -3

echo "== train Whisper (tiny)"
python src/train_whisper.py --train_dir "$S/data/kathbath_train" --val_dir "$S/data/kathbath_val" \
  --model "$S/models/tiny_whisper" --out "$S/runs/tiny_whisper" --lr 1e-3 --max_steps ${WHISPER_SMOKE_STEPS:-2500} \
  --warmup_steps 100 --bs 16 --accum 1 --eval_bs 16 --eval_steps 500 --logging_steps 500 --workers 0 \
  --no_grad_ckpt > "$S/whisper.log" 2>&1 || { tail -40 "$S/whisper.log"; exit 1; }

echo "== evaluate + report"
python src/evaluate.py --data "$S/data" --out "$S/results" \
  --model "tiny_whisper=whisper:$S/runs/tiny_whisper" --model "tiny_ctc=ctc:$S/runs/tiny_ctc" 2>&1 | grep -E "evaluate|FAIL|Error" || true
python src/report.py --results "$S/results" --data "$S/data" --runs "$S/runs" --boot 200
echo "SMOKE TEST FINISHED - inspect the table above: CTC WER should be far below 100% on the synthetic sets."
