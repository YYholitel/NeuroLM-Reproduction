#!/usr/bin/env bash
set -euo pipefail
JOB="$1"; SEED="$2"; ROOT="$(dirname "$(dirname "$JOB")")"; BASE="$(dirname "$ROOT")"
PY="$BASE/envs/neurolm-py311/bin/python"
[[ "${WORLD_SIZE:-}" == 1 && "${GPU_NUM_PER_NODE:-}" == 2 && "${RANK:-}" == 0 ]]
mkdir "$JOB/train"
exec > >(tee "$JOB/worker.log") 2>&1
MONITOR=''
finish() { rc=$?; trap - EXIT; if [[ -n "$MONITOR" ]]; then kill "$MONITOR" 2>/dev/null || true; wait "$MONITOR" 2>/dev/null || true; fi; printf '%s\n' "$rc" > "$JOB/exit_code.txt"; exit "$rc"; }
trap finish EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TIKTOKEN_CACHE_DIR="$BASE/cache/tiktoken" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
TMPDIR="$(mktemp -d /tmp/neurolm-official-s3.XXXXXXXX)";export TMPDIR
"$PY" "$ROOT/scripts/preflight_official_stage3.py" --job "$JOB" --worker
"$PY" "$ROOT/scripts/monitor_vq_gpu.py" --output "$JOB/gpu_metrics.csv" & MONITOR=$!
ln -s "$ROOT/assets/text" "$JOB/train/text"
cd "$ROOT/author"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
  train_instruction.py --datasets TUAB TUEV HMC Workload TUSL --dataset_dir "$ROOT/data" \
  --out_dir "$JOB/train" --NeuroLM_path "$JOB/../stage2/run/checkpoints/NeuroLM-B/ckpt.pt" \
  --tokenizer_path "$BASE/checkpoints/VQ.pt" --seed "$SEED" --epochs 5 --save_ckpt_freq 5 \
  --num_workers 4 --skip_evaluation --eeg_batch_size 32 --text_batch_size 8 --gradient_accumulation_steps 8 \
  --learning_rate 5e-4 --min_lr 5e-5 --warmup_ratio .1 --warmup_epochs 1 --weight_decay .1 --grad_clip 1 --log_interval 10
"$PY" "$ROOT/scripts/audit_official_checkpoint.py" --job "$JOB" --stage 3
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
  "$ROOT/scripts/eval_downstream.py" --author-dir "$ROOT/author" \
  --checkpoint "$JOB/train/checkpoints/instruction-B/ckpt.pt" --dataset-dir "$ROOT/data" --out-dir "$JOB/eval" \
  --tasks TUSL TUAB TUEV HMC Workload --mode constrained --batch-size 32 --num-workers 4 --device cuda --precision bfloat16
"$PY" "$ROOT/scripts/audit_downstream_eval.py" --run-dir "$JOB/eval" --output "$JOB/eval_audit.json"
printf 'STAGE3_FIVE_EPOCH_COMPLETE seed=%s\n' "$SEED"
