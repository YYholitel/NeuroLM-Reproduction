#!/usr/bin/env bash
set -euo pipefail
JOB="$1"; SEED="$2"; ROOT="$(dirname "$(dirname "$JOB")")"; BASE="$(dirname "$ROOT")"
[[ "${WORLD_SIZE:-}" == 2 && "${GPU_NUM_PER_NODE:-}" == 8 && "${RANK:-}" =~ ^[01]$ ]]
mkdir "$JOB/node-$RANK"; mkdir -p "$JOB/run"
exec > >(tee -a "$JOB/run/pretrain.log" "$JOB/node-$RANK/worker.log") 2>&1
MONITOR=''
finish() { rc=$?; trap - EXIT; if [[ -n "$MONITOR" ]]; then kill "$MONITOR" 2>/dev/null || true; wait "$MONITOR" 2>/dev/null || true; fi; printf '%s\n' "$rc" > "$JOB/node-$RANK/exit_code.txt"; exit "$rc"; }
trap finish EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TORCH_NCCL_ASYNC_ERROR_HANDLING=1
printf 'STAGE2_SEED=%s NODE=%s HOST=%s\n' "$SEED" "$RANK" "$(hostname)"
python3 "$ROOT/scripts/monitor_vq_gpu.py" --output "$JOB/node-$RANK/gpu_metrics.csv" & MONITOR=$!
python3 "$ROOT/scripts/preflight_official_tusl3.py" --root "$ROOT" --worker
cd "$ROOT/code"
python3 -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --max_restarts=0 \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --node_rank="$RANK" \
  train_pretrain.py --dataset_dir "$ROOT/assets" --train_manifest "$ROOT/assets/train_files.txt" \
  --data_ready_manifest "$ROOT/assets/data_ready.json" --text_dir "$ROOT/assets/text" --out_dir "$JOB/run" \
  --tokenizer_path "$BASE/checkpoints/VQ.pt" --initial_neurolm_path "$BASE/checkpoints/NeuroLM-B.pt" \
  --epochs 20 --warmup_epochs 2 --max_updates 19140 --eeg_batch_size 30 --text_batch_size 2 \
  --gradient_accumulation_steps 1 --num_workers 2 --learning_rate 6e-4 --min_lr 6e-5 \
  --weight_decay .1 --beta1 .9 --beta2 .95 --grad_clip 1 --log_interval 10 --seed "$SEED"
if [[ "$RANK" == 0 ]]; then python3 "$ROOT/scripts/audit_official_checkpoint.py" --job "$JOB" --stage 2; fi
