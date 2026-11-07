#!/usr/bin/env bash
set -euo pipefail
: "${VQ_SMOKE_ROOT:?Set the isolated smoke directory}"
: "${WORLD_SIZE:?Missing TI-ONE node count}"
: "${RANK:?Missing TI-ONE node rank}"
: "${GPU_NUM_PER_NODE:?Missing TI-ONE GPU count}"
: "${MASTER_ADDR:?Missing rendezvous host}"
: "${MASTER_PORT:?Missing rendezvous port}"
NNODES="$WORLD_SIZE"
NODE_RANK="$RANK"
if [[ "$NNODES" != 2 || "$GPU_NUM_PER_NODE" != 8 || ! "$NODE_RANK" =~ ^[01]$ ]]; then
    echo 'Expected two nodes with eight GPUs each.'
    exit 2
fi
ROOT="$(realpath "$VQ_SMOKE_ROOT")"
RUN="$ROOT/run"
mkdir -p "$RUN"
mkdir "$RUN/node-$NODE_RANK"
exec > >(tee "$RUN/node-$NODE_RANK/worker.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/node-'"$NODE_RANK"'/exit_code.txt"' EXIT
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="sts.info,metadata.tencentyun.com,169.254.0.0/16"
export NO_PROXY="$no_proxy"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python3 - <<'PY'
import torch, sys, json
assert torch.cuda.device_count() == 8
print(json.dumps(dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                     gpu_count=torch.cuda.device_count(), wandb=False)), flush=True)
PY
cd "$ROOT/code"
timeout --signal=TERM --kill-after=30s 1200s python3 -m torch.distributed.run \
    --nnodes=2 --nproc_per_node=8 \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --node_rank="$NODE_RANK" \
    train_vq.py \
    --dataset_dir "$ROOT/assets/repeated_diagnostic" \
    --out_dir "$RUN" \
    --gpt2_embedding_path "$ROOT/assets/gpt2_wte.pt" \
    --epochs 1 --warmup_epochs 0 \
    --batch_size 32 --text_batch_size 8 --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 --min_lr 1e-5 --log_interval 1 --seed 42
