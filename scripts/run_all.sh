#!/usr/bin/env bash
# Full pipeline. Usage:
#   bash scripts/run_all.sh quick     # ~10 h of data, short training: validates the whole pipeline on a real GPU
#   bash scripts/run_all.sh full      # ~100 h of data, full training (the real experiment)
# Individual stages:  bash scripts/run_all.sh full prepare|train_whisper|train_ctc|eval|report
# Everything is resumable: prepare skips nothing (re-run overwrites), training auto-resumes from the
# last checkpoint, evaluation skips (model,set) pairs that already have predictions.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE=${1:-quick}
STAGE=${2:-all}
mkdir -p logs

if [[ "$PROFILE" == "quick" ]]; then
  TRAIN_HOURS=${TRAIN_HOURS:-10};  TEST_UTTS=${TEST_UTTS:-300}; IV_UTTS=${IV_UTTS:-200}
  W_STEPS=${W_STEPS:-1200}; W_EVAL=${W_EVAL:-400}; W_WARM=${W_WARM:-100}
  C_STEPS=${C_STEPS:-3000}; C_EVAL=${C_EVAL:-1000}; C_WARM=${C_WARM:-300}
  RUNS=runs_quick; RES=results_quick; DATA=${DATA:-data_quick}
elif [[ "$PROFILE" == "full" ]]; then
  TRAIN_HOURS=${TRAIN_HOURS:-100}; TEST_UTTS=${TEST_UTTS:-2000}; IV_UTTS=${IV_UTTS:-1500}
  W_STEPS=${W_STEPS:-8000}; W_EVAL=${W_EVAL:-1000}; W_WARM=${W_WARM:-500}
  C_STEPS=${C_STEPS:-12000}; C_EVAL=${C_EVAL:-1000}; C_WARM=${C_WARM:-1000}
  RUNS=runs; RES=results; DATA=${DATA:-data}
else
  echo "profile must be quick|full"; exit 1
fi
W_BS=${W_BS:-16}; W_ACC=${W_ACC:-2}      # effective batch 32
C_BS=${C_BS:-8};  C_ACC=${C_ACC:-4}      # effective batch 32
CV_ARGS=()
[[ -n "${COMMONVOICE_DIR:-}" ]] && CV_ARGS=(--commonvoice_dir "$COMMONVOICE_DIR")

run() { echo -e "\n>>> $*"; "$@"; }

stage_prepare() {
  run python src/check_env.py --path . ${ALLOW_CPU:+--allow_cpu}
  run python src/textnorm.py
  run python tests/test_metrics.py
  run python src/probe_datasets.py
  run python src/prepare_data.py --out "$DATA" --train_hours "$TRAIN_HOURS" --test_utts "$TEST_UTTS" \
      --indicvoices_utts "$IV_UTTS" ${CV_ARGS[@]+"${CV_ARGS[@]}"} 2>&1 | tee logs/prepare_${PROFILE}.log
}
stage_train_whisper() {
  run python src/train_whisper.py --train_dir "$DATA/kathbath_train" --val_dir "$DATA/kathbath_val" \
      --model "${WHISPER_BASE:-openai/whisper-small}" --out "$RUNS/whisper_small" \
      --max_steps "$W_STEPS" --eval_steps "$W_EVAL" --warmup_steps "$W_WARM" --bs "$W_BS" --accum "$W_ACC" \
      2>&1 | tee logs/train_whisper_${PROFILE}.log
}
stage_train_ctc() {
  run python src/train_ctc.py --train_dir "$DATA/kathbath_train" --val_dir "$DATA/kathbath_val" \
      --model "${CTC_BASE:-facebook/mms-300m}" --out "$RUNS/mms300m_ctc" \
      --max_steps "$C_STEPS" --eval_steps "$C_EVAL" --warmup_steps "$C_WARM" --bs "$C_BS" --accum "$C_ACC" \
      2>&1 | tee logs/train_ctc_${PROFILE}.log
}
stage_eval() {
  run python src/evaluate.py --data "$DATA" --out "$RES" \
      --model "whisper_small_zeroshot=whisper:${WHISPER_BASE:-openai/whisper-small}" \
      --model "whisper_small_finetuned=whisper:$RUNS/whisper_small" \
      --model "mms300m_ctc_finetuned=ctc:$RUNS/mms300m_ctc" 2>&1 | tee logs/eval_${PROFILE}.log
  if [[ "${INCLUDE_INDICCONFORMER:-0}" == "1" ]]; then
    # Optional zero-shot reference; gated + remote code. A failure here must not kill the run.
    python src/evaluate.py --data "$DATA" --out "$RES" --max_utts "${IC_MAX_UTTS:-500}" \
      --model "indicconformer_600m_zeroshot=indicconformer:ai4bharat/indic-conformer-600m-multilingual" \
      2>&1 | tee logs/eval_indicconformer_${PROFILE}.log || echo "!! IndicConformer evaluation failed (optional)"
  fi
}
stage_report() {
  run python src/report.py --results "$RES" --data "$DATA" --runs "$RUNS" \
      --order whisper_small_zeroshot whisper_small_finetuned mms300m_ctc_finetuned indicconformer_600m_zeroshot
}

case "$STAGE" in
  all) stage_prepare; stage_train_whisper; stage_train_ctc; stage_eval; stage_report ;;
  prepare) stage_prepare ;;
  train_whisper) stage_train_whisper ;;
  train_ctc) stage_train_ctc ;;
  eval) stage_eval ;;
  report) stage_report ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
