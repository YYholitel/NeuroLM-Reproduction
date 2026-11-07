#!/usr/bin/env bash
set -euo pipefail
: "${VQ_FULL_ROOT:?Set the isolated full-training directory}"
: "${WORLD_SIZE:?Missing TI-ONE node count}"
: "${RANK:?Missing TI-ONE node rank}"
: "${GPU_NUM_PER_NODE:?Missing TI-ONE GPU count}"
: "${MASTER_ADDR:?Missing rendezvous host}"
: "${MASTER_PORT:?Missing rendezvous port}"
if [[ "$WORLD_SIZE" != 2 || "$GPU_NUM_PER_NODE" != 8 || ! "$RANK" =~ ^[01]$ ]]; then
    echo 'Expected two nodes with eight GPUs each.'
    exit 2
fi
ROOT="$(realpath "$VQ_FULL_ROOT")"
RUN="$ROOT/run"
mkdir -p "$RUN"
mkdir "$RUN/node-$RANK"
exec > >(tee "$RUN/node-$RANK/worker.log") 2>&1
GPU_MONITOR_PID=''
finish() {
    rc=$?
    trap - EXIT
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    printf '%s\n' "$rc" > "$RUN/node-$RANK/exit_code.txt"
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
python3 "$ROOT/gpu_monitor.py" --output "$RUN/node-$RANK/gpu_metrics.csv" &
GPU_MONITOR_PID=$!
python3 - <<'PY'
import json,os,sys,torch
from pathlib import Path
root=Path(os.environ['VQ_FULL_ROOT'])
ready=json.loads((root/'assets/data_ready.json').read_text())
assert ready['ready'] and ready['total_samples']==459656
assert torch.cuda.device_count()==8
assert (root/'run/text/train.bin').stat().st_size==18071164978
assert (root/'run/text/val.bin').stat().st_size==8869212
print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                     gpu_count=8,wandb=False,plan_sha256=ready['plan_sha256'])),flush=True)
PY
cd "$ROOT/code"
python3 -m torch.distributed.run \
    --nnodes=2 --nproc_per_node=8 \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --node_rank="$RANK" \
    train_vq.py \
    --dataset_dir "$ROOT/assets" \
    --train_manifest "$ROOT/assets/train_files.txt" \
    --data_ready_manifest "$ROOT/assets/data_ready.json" \
    --out_dir "$RUN" \
    --gpt2_embedding_path "$ROOT/assets/gpt2_wte.pt" \
    --epochs 50 --warmup_epochs 5 \
    --batch_size 32 --text_batch_size 8 --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 --min_lr 1e-5 --weight_decay 1e-4 \
    --beta1 0.9 --beta2 0.999 --grad_clip 0 --log_interval 10 --seed 42
