#!/usr/bin/env bash
# Invoke inside a TI-ONE worker with the relocated, verified environment.
set -euo pipefail
: "${NEUROLM_ROOT:?Set the private CFS project root}"
: "${NEUROLM_RUN_ID:?Set a unique run ID}"
ROOT="$(realpath "$NEUROLM_ROOT")"
RUN="$ROOT/outputs/$NEUROLM_RUN_ID"
ENV="$ROOT/envs/neurolm-py311"
PY="$ENV/bin/python"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/worker.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/exit_code.txt"' EXIT
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="sts.info,metadata.tencentyun.com,169.254.0.0/16"
export NO_PROXY="$no_proxy"
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TIKTOKEN_CACHE_DIR="$ROOT/cache/tiktoken"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"

# Refuse an accidental restart into a previous epoch/checkpoint.
if [[ -e "$RUN/checkpoints" || -e "$RUN/completed.json" ]]; then
    echo 'Existing training output; choose a new run ID or explicitly handle resume.'
    exit 2
fi
date -u
nvidia-smi
"$PY" - <<'PY'
import sys, torch, numpy
assert sys.version_info[:3] == (3, 11, 8), sys.version
assert torch.__version__.split('+')[0] == '2.2.2', torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
assert numpy.__version__ == '1.26.4', numpy.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
assert torch.cuda.is_bf16_supported()
print('Validated runtime:', sys.version, torch.__version__, torch.version.cuda)
PY
"$PY" -m pip freeze > "$RUN/pip-freeze.txt"
"$PY" "$ROOT/code/reproduction/scripts/preflight_instruction_run.py" \
    --data-dir "$ROOT/data" --checkpoint-dir "$ROOT/checkpoints" --output "$RUN/input_preflight.json"
ln -s "$ROOT/data/text" "$RUN/text"
cd "$ROOT/code/author"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 train_instruction.py \
    --datasets TUAB TUEV HMC Workload TUSL \
    --dataset_dir "$ROOT/data" --out_dir "$RUN" \
    --NeuroLM_path "$ROOT/checkpoints/NeuroLM-B.pt" \
    --tokenizer_path "$ROOT/checkpoints/VQ.pt" \
    --seed 42 --epochs 1 --save_ckpt_freq 1 --num_workers 4 \
    --eeg_batch_size 64 --text_batch_size 16 --gradient_accumulation_steps 1 \
    --learning_rate 5e-4 --min_lr 5e-5 --warmup_ratio 0.1 --warmup_epochs 1 \
    --weight_decay 0.1 --grad_clip 1 --log_interval 10
"$PY" "$ROOT/code/reproduction/scripts/summarize_instruction_run.py" --run-dir "$RUN"
