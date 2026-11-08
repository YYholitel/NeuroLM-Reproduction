#!/usr/bin/env bash
# Run the full five-task instruction epoch on one TI-ONE worker with two H20 GPUs.
set -euo pipefail

# Platform allocates an isolated single worker with two H20 GPUs.
: "${WORLD_SIZE:?TI-ONE must provide the node count}"
: "${RANK:?TI-ONE must provide the node rank}"
: "${GPU_NUM_PER_NODE:?TI-ONE must provide the GPU count}"
: "${NEUROLM_CODE_ROOT:?Set the frozen code-bundle root}"
if [[ "$WORLD_SIZE" != 1 || "$RANK" != 0 || "$GPU_NUM_PER_NODE" != 2 ]]; then
    echo 'Expected one TI-ONE node with two H20 GPUs.'
    exit 2
fi
CODE_ROOT="$(realpath "$NEUROLM_CODE_ROOT")"

: "${NEUROLM_ROOT:?Set the private project root on the training server}"
: "${NEUROLM_RUN_ID:?Set a unique run ID}"

ROOT="$(realpath "$NEUROLM_ROOT")"
RUN="$ROOT/outputs/$NEUROLM_RUN_ID"
ENV="$ROOT/envs/neurolm-py311"
PY="$ENV/bin/python"

mkdir -p "$ROOT/outputs"
mkdir "$RUN" # Refuse to overwrite or silently resume an earlier experiment.
exec > >(tee -a "$RUN/worker.log") 2>&1
record_exit() {
    local rc=$?
    printf '%s\n' "$rc" > "$RUN/exit_code.txt.tmp"
    mv "$RUN/exit_code.txt.tmp" "$RUN/exit_code.txt"
}
trap record_exit EXIT

export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TIKTOKEN_CACHE_DIR="$ROOT/cache/tiktoken"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# DataLoader IPC needs short, container-local paths; long run paths exceed
# AF_UNIX limits, and NFS sockets cause busy-file errors during worker cleanup.
# Persistent experiment inputs, logs, and checkpoints remain under ROOT.
TMPDIR="$(mktemp -d /tmp/neurolm-i.XXXXXXXX)"
export TMPDIR

date -u
check_gpu_idle() {
    local memory_csv compute_pids mib
    local -a free_mib
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
    memory_csv="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
    mapfile -t free_mib <<< "$memory_csv"
    if [[ ${#free_mib[@]} -ne 2 ]]; then
        echo "Expected exactly two GPUs; found ${#free_mib[@]}."
        exit 2
    fi
    for mib in "${free_mib[@]}"; do
        if [[ ! "$mib" =~ ^[0-9]+$ ]] || (( mib < 90000 )); then
            echo "GPU free-memory check failed ($mib MiB); wait until both H20 GPUs are idle."
            exit 2
        fi
    done
    compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
    if [[ -n "${compute_pids//[[:space:]]/}" ]]; then
        echo 'Existing GPU compute processes detected. Refusing to start this experiment.'
        exit 2
    fi
}
check_gpu_idle
"$PY" - <<'PY'
import sys
import numpy
import torch

assert sys.version_info[:3] == (3, 11, 8), sys.version
assert torch.__version__.split('+')[0] == '2.2.2', torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
assert numpy.__version__ == '1.26.4', numpy.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
assert torch.cuda.is_bf16_supported()
for index in range(2):
    assert 'H20' in torch.cuda.get_device_name(index), torch.cuda.get_device_name(index)
print('Validated runtime:', sys.version, torch.__version__, torch.version.cuda)
PY

"$PY" -m pip freeze > "$RUN/pip-freeze.txt"
"$PY" "$CODE_ROOT/reproduction/scripts/preflight_instruction_run.py" \
    --data-dir "$ROOT/data" --checkpoint-dir "$ROOT/checkpoints" \
    --output "$RUN/input_preflight.json"
ln -s "$ROOT/data/text" "$RUN/text"

# Input hashing can take time; recheck for jobs that started during preflight.
check_gpu_idle
cd "$CODE_ROOT/author"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
    train_instruction.py \
    --datasets TUAB TUEV HMC Workload TUSL \
    --dataset_dir "$ROOT/data" --out_dir "$RUN" \
    --NeuroLM_path "$ROOT/checkpoints/NeuroLM-B.pt" \
    --tokenizer_path "$ROOT/checkpoints/VQ.pt" \
    --seed 42 --epochs 1 --save_ckpt_freq 1 --num_workers 4 \
    --eeg_batch_size 32 --text_batch_size 8 --gradient_accumulation_steps 8 \
    --learning_rate 5e-4 --min_lr 5e-5 --warmup_ratio 0.1 --warmup_epochs 1 \
    --weight_decay 0.1 --grad_clip 1 --log_interval 10

"$PY" "$CODE_ROOT/reproduction/scripts/summarize_instruction_run.py" --run-dir "$RUN"
