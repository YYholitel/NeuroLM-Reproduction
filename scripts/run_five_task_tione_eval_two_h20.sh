#!/usr/bin/env bash
# Recover evaluation from a completed epoch without taking optimizer steps.
set -euo pipefail
: "${NEUROLM_ROOT:?Set the private project root}"
: "${NEUROLM_CODE_ROOT:?Set the frozen code-bundle root}"
: "${NEUROLM_RUN_ID:?Set a new evaluation run ID}"
: "${NEUROLM_TRAIN_RUN:?Set the completed training output directory}"
: "${NEUROLM_VERIFIED_INPUT_RUN:?Set the prior run with a passing full input preflight}"
[[ "${WORLD_SIZE:?}" == 1 && "${RANK:?}" == 0 && "${GPU_NUM_PER_NODE:?}" == 2 ]]
ROOT="$(realpath "$NEUROLM_ROOT")"
CODE_ROOT="$(realpath "$NEUROLM_CODE_ROOT")"
TRAIN_RUN="$(realpath "$NEUROLM_TRAIN_RUN")"
RUN="$ROOT/outputs/$NEUROLM_RUN_ID"
PY="$ROOT/envs/neurolm-py311/bin/python"
mkdir "$RUN"
exec > >(tee -a "$RUN/worker.log") 2>&1
record_exit() { local rc=$?; printf '%s\n' "$rc" > "$RUN/exit_code.txt"; }
trap record_exit EXIT
export PATH="$ROOT/envs/neurolm-py311/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=4
export TIKTOKEN_CACHE_DIR="$ROOT/cache/tiktoken"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Preserve the original symbolic backend. The crash was reproduced in the
# BF16 LM-head for small batch shapes; this bundle carries the padding patch.
unset SYMPY_GROUND_TYPES
ulimit -c 0
TMPDIR="$(mktemp -d /tmp/neurolm-e.XXXXXXXX)"
export TMPDIR

"$PY" - "$TRAIN_RUN" "$RUN" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy, sympy, torch
from sympy.external.gmpy import GROUND_TYPES
train, run = map(Path, sys.argv[1:])
assert sys.version_info[:3] == (3, 11, 8)
assert torch.__version__.split('+')[0] == '2.2.2' and torch.version.cuda == '12.1'
assert numpy.__version__ == '1.26.4'
pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
assert not pids.strip(), 'Existing GPU jobs'
free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
assert len(free.splitlines()) == 2 and all(int(x) >= 90000 for x in free.splitlines())
assert torch.cuda.device_count() == 2 and torch.cuda.is_bf16_supported()
assert all('H20' in torch.cuda.get_device_name(i) for i in range(2))
ck = train / 'checkpoints/instruction-B/ckpt.pt'
with ck.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
state = torch.load(ck, map_location='cpu', weights_only=False)
steps = sorted(set(int(x['step']) for x in state['optimizer']['state'].values()))
assert state['epoch'] == 0 and state['iter_num'] == 7182 and steps == [898]
record = {'source_training_run': train.name, 'checkpoint_sha256': digest,
          'checkpoint_bytes': ck.stat().st_size, 'epoch': state['epoch'],
          'iter_num': state['iter_num'], 'optimizer_steps': steps,
          'optimizer_states': len(state['optimizer']['state']),
          'training_worker_exit_code': int((train / 'exit_code.txt').read_text()),
          'training_complete': True, 'evaluation_only': True,
          'sympy': sympy.__version__, 'sympy_ground_types': GROUND_TYPES,
          'inference_lm_head_min_batch': 48}
(run / 'training_provenance.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
PY
"$PY" -m pip freeze > "$RUN/pip-freeze.txt"
"$PY" - "$NEUROLM_VERIFIED_INPUT_RUN" "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
source, run = map(Path, sys.argv[1:])
path = source / 'input_preflight.json'
raw = path.read_bytes()
record = json.loads(raw)
assert record['passed'] and not record['errors']
assert sum(x['actual'] for x in record['counts'].values()) == 660896
assert all(x['actual'] == x['expected'] for x in record['counts'].values())
record['reused_verified_input_manifest'] = {'run': source.name, 'sha256': hashlib.sha256(raw).hexdigest()}
# These fixed inputs were not changed during recovery. The author loaders
# enumerate splits again; their exact counts are checked by the final audit.
(run / 'input_preflight.json').write_text(json.dumps(record, indent=2) + '\n')
PY
"$PY" "$CODE_ROOT/reproduction/scripts/verify_lm_head_padding.py" --output "$RUN/lm_head_padding_probe.json"
ln -s "$ROOT/data/text" "$RUN/text"
mkdir "$RUN/checkpoints"
ln -s "$TRAIN_RUN/checkpoints/instruction-B" "$RUN/checkpoints/instruction-B"
cd "$CODE_ROOT/author"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
    train_instruction.py --eval_only \
    --datasets TUAB TUEV HMC Workload TUSL \
    --dataset_dir "$ROOT/data" --out_dir "$RUN" \
    --NeuroLM_path "$ROOT/checkpoints/NeuroLM-B.pt" --tokenizer_path "$ROOT/checkpoints/VQ.pt" \
    --seed 42 --epochs 1 --save_ckpt_freq 1 --num_workers 4 \
    --eeg_batch_size 32 --text_batch_size 8 --gradient_accumulation_steps 8 \
    --learning_rate 5e-4 --min_lr 5e-5 --warmup_ratio 0.1 --warmup_epochs 1 \
    --weight_decay 0.1 --grad_clip 1 --log_interval 10
"$PY" "$CODE_ROOT/reproduction/scripts/summarize_instruction_run.py" --run-dir "$RUN"
"$PY" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
source = json.loads((run / 'training_provenance.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
assert source['checkpoint_sha256'] == summary['checkpoint_sha256'], 'Evaluation changed the checkpoint'
summary['training_provenance'] = source
(run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
PY
