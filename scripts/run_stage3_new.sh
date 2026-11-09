#!/usr/bin/env bash
# Matched one-epoch instruction tuning, then identical five-task constrained evaluation.
set -euo pipefail
BASE=/mnt/cfs-omni/lhy/neurolm-reproduction
ROOT="$BASE/stage3-new-pretrain-20261109-r1"
PY="$BASE/envs/neurolm-py311/bin/python"
NEW="$BASE/pretrain-full20-20261109-r1/run/checkpoints/NeuroLM-B/ckpt.pt"
OLD="$BASE/outputs/neurolm-instruction-five-e1-2gpu-20261108-r2/checkpoints/instruction-B/ckpt.pt"
[[ "${WORLD_SIZE:-}" == 1 && "${GPU_NUM_PER_NODE:-}" == 2 && "${RANK:-}" == 0 ]]
mkdir "$ROOT/train"
exec > >(tee "$ROOT/worker.log") 2>&1
GPU_MONITOR_PID=''
finish() {
    rc=$?;trap - EXIT
    if [[ -n "$GPU_MONITOR_PID" ]]; then kill "$GPU_MONITOR_PID" 2>/dev/null || true; wait "$GPU_MONITOR_PID" 2>/dev/null || true; fi
    printf '%s\n' "$rc" > "$ROOT/exit_code.txt"
    exit "$rc"
}
trap finish EXIT
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TIKTOKEN_CACHE_DIR="$BASE/cache/tiktoken" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
TMPDIR="$(mktemp -d /tmp/neurolm-s3.XXXXXXXX)";export TMPDIR
date -u
"$PY" - "$ROOT" <<'CHECK'
import sys,json,torch,numpy,hashlib,subprocess
from pathlib import Path
r=Path(sys.argv[1]);assert json.loads((r/'preflight.json').read_text())['ok']
assert sys.version_info[:3]==(3,11,8) and torch.__version__.split('+')[0]=='2.2.2'
assert torch.version.cuda=='12.1' and numpy.__version__=='1.26.4'
assert torch.cuda.device_count()==2 and torch.cuda.is_bf16_supported()
for name,h in json.loads((r/'author/bundle_manifest.json').read_text()).items():
    assert hashlib.sha256((r/'author'/name).read_bytes()).hexdigest()==h,name
rows=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
assert len(rows)==2 and all('H20' in row and int(row.rsplit(',',1)[1])<2048 for row in rows)
print('WORKER_PREFLIGHT='+json.dumps(dict(ok=True,gpus=rows,torch=torch.__version__)),flush=True)
CHECK
"$PY" "$ROOT/scripts/monitor_vq_gpu.py" --output "$ROOT/gpu_metrics.csv" &
GPU_MONITOR_PID=$!
ln -s "$BASE/data/text" "$ROOT/train/text"
cd "$ROOT/author"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
    train_instruction.py --datasets TUAB TUEV HMC Workload TUSL \
    --dataset_dir "$BASE/data" --out_dir "$ROOT/train" --NeuroLM_path "$NEW" \
    --tokenizer_path "$BASE/vq-fixed-e5-20261108-r1/run/checkpoints/VQ/ckpt-4.pt" \
    --seed 42 --epochs 1 --save_ckpt_freq 1 --num_workers 4 --skip_evaluation \
    --eeg_batch_size 32 --text_batch_size 8 --gradient_accumulation_steps 8 \
    --learning_rate 5e-4 --min_lr 5e-5 --warmup_ratio 0.1 --warmup_epochs 1 \
    --weight_decay 0.1 --grad_clip 1 --log_interval 10
"$PY" "$ROOT/audit_checkpoint.py"
for VARIANT in new official; do
    if [[ "$VARIANT" == new ]]; then CHECKPOINT="$ROOT/train/checkpoints/instruction-B/ckpt.pt"; else CHECKPOINT="$OLD"; fi
    printf 'EVALUATING_VARIANT=%s\n' "$VARIANT"
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
        "$ROOT/scripts/eval_downstream.py" --author-dir "$ROOT/author" --checkpoint "$CHECKPOINT" \
        --dataset-dir "$BASE/data" --out-dir "$ROOT/eval-$VARIANT" \
        --tasks TUSL TUAB TUEV HMC Workload --mode constrained --batch-size 32 --num-workers 4 \
        --device cuda --precision bfloat16
    "$PY" "$ROOT/scripts/audit_downstream_eval.py" --run-dir "$ROOT/eval-$VARIANT" --output "$ROOT/audit-$VARIANT.json"
done
"$PY" "$ROOT/compare.py"
printf 'STAGE3_MATCHED_COMPARISON_COMPLETE\n'
