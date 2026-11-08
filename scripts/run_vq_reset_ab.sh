#!/usr/bin/env bash
set -euo pipefail
: "${VQ_AB_ROOT:?Set isolated A/B root}"
: "${WORLD_SIZE:?Missing node count}"
: "${RANK:?Missing node rank}"
: "${GPU_NUM_PER_NODE:?Missing GPU count}"
: "${MASTER_ADDR:?Missing rendezvous address}"
: "${MASTER_PORT:?Missing rendezvous port}"
[[ "$WORLD_SIZE" == 2 && "$GPU_NUM_PER_NODE" == 8 && "$RANK" =~ ^[01]$ ]]
[[ "$MASTER_PORT" -lt 65535 ]]
ROOT="$(realpath "$VQ_AB_ROOT")"
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
ARM_INDEX=0
for ARM in reset_off reset_on; do
    if [[ "$ARM" == reset_off ]]; then PATIENCE=0; else PATIENCE=100; fi
    RUN="$ROOT/runs/$ARM"
    [[ ! -e "$RUN/completed.json" && ! -e "$RUN/checkpoints/VQ/ckpt.pt" ]]
    printf 'AB_ARM_START=%s rank=%s\n' "$ARM" "$RANK"
    python3 -m torch.distributed.run \
        --nnodes=2 --nproc_per_node=8 \
        --master_addr="$MASTER_ADDR" --master_port="$((MASTER_PORT + ARM_INDEX))" --node_rank="$RANK" \
        train_vq.py \
        --dataset_dir "$ROOT/assets" \
        --train_manifest "$ROOT/assets/train_files.txt" \
        --data_ready_manifest "$ROOT/assets/data_ready.json" \
        --out_dir "$RUN" --gpt2_embedding_path "$ROOT/assets/gpt2_wte.pt" \
        --epochs 50 --warmup_epochs 5 --max_updates 500 --usage_window 50 \
        --batch_size 32 --text_batch_size 8 --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 --min_lr 1e-5 --weight_decay 1e-4 \
        --beta1 0.9 --beta2 0.999 --grad_clip 0 --log_interval 10 --seed 42 \
        --vq_beta 1 --vq_ema_decay 0.99 --vq_dead_code_patience "$PATIENCE" \
        --vq_dead_code_interval 50 --vq_init_samples_per_rank 2048 \
        2>&1 | tee "$ROOT/node-$RANK/$ARM.log"
    # Both torchrun groups finish their final barrier before the next arm.
    python3 - "$RUN" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
done=json.loads((run/'completed.json').read_text())
assert done['steps']==500 and done['world_size']==16,done
assert len((run/'metrics.jsonl').read_text().splitlines())==500
print('AB_ARM_DONE='+json.dumps(done),flush=True)
PY
    ARM_INDEX=$((ARM_INDEX + 1))
done
if [[ "$RANK" == 0 ]]; then python3 "$ROOT/verify_results.py" --root "$ROOT"; fi
