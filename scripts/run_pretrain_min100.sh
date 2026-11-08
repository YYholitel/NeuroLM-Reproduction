#!/usr/bin/env bash
set -euo pipefail
: "${PRETRAIN_MIN_ROOT:?Set isolated A/B root}"
: "${WORLD_SIZE:?Missing node count}"
: "${RANK:?Missing node rank}"
: "${GPU_NUM_PER_NODE:?Missing GPU count}"
: "${MASTER_ADDR:?Missing rendezvous address}"
: "${MASTER_PORT:?Missing rendezvous port}"
[[ "$WORLD_SIZE" == 2 && "$GPU_NUM_PER_NODE" == 8 && "$RANK" =~ ^[01]$ ]]
[[ "$MASTER_PORT" -lt 65535 ]]
ROOT="$(realpath "$PRETRAIN_MIN_ROOT")"
mkdir "$ROOT/node-$RANK"
exec > >(tee "$ROOT/node-$RANK/worker.log") 2>&1
GPU_MONITOR_PID=''
finish() {
    rc=$?
    trap - EXIT
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    printf '%s\n' "$rc" > "$ROOT/node-$RANK/exit_code.txt"
    exit "$rc"
}
trap finish EXIT
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="sts.info,metadata.tencentyun.com,169.254.0.0/16"
export NO_PROXY="$no_proxy"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python3 "$ROOT/gpu_monitor.py" --output "$ROOT/node-$RANK/gpu_metrics.csv" &
GPU_MONITOR_PID=$!
python3 "$ROOT/preflight.py" --worker
cd "$ROOT/code"
RUN="$ROOT/run"
python3 -m torch.distributed.run \
    --nnodes=2 --nproc_per_node=8 \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --node_rank="$RANK" \
    train_pretrain.py \
    --dataset_dir "$ROOT/assets" --train_manifest "$ROOT/assets/train_files.txt" \
    --data_ready_manifest "$ROOT/assets/data_ready.json" \
    --validation_manifest "$ROOT/probe/validation_manifest.json" \
    --text_dir "$ROOT/assets/text" --out_dir "$RUN" \
    --tokenizer_path /mnt/cfs-omni/lhy/neurolm-reproduction/vq-fixed-e5-20261108-r1/run/checkpoints/VQ/ckpt-4.pt \
    --gpt2_init_dir "$ROOT/gpt2" \
    --epochs 20 --warmup_epochs 2 --max_updates 100 --eval_interval 50 --val_text_batches 4 \
    --eeg_batch_size 30 --text_batch_size 2 --gradient_accumulation_steps 1 --num_workers 2 \
    --learning_rate 6e-4 --min_lr 6e-5 --weight_decay 0.1 \
    --beta1 0.9 --beta2 0.95 --grad_clip 1 --log_interval 10 --seed 42
python3 - "$RUN" <<'PYDONE'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);done=json.loads((run/'completed.json').read_text())
assert done['steps']==100 and done['world_size']==16 and done['frozen_state_unchanged'] and done['gpt_weights_updated']
assert len((run/'metrics.jsonl').read_text().splitlines())==100
print('PRETRAIN_MIN_DONE='+json.dumps(done),flush=True)
PYDONE
