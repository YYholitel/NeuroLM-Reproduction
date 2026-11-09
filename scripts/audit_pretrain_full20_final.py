"""Read-only verification of our completed, trusted stage-two checkpoint and logs."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import statistics
import torch

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-full20-20261109-r1')
checkpoint=ROOT/'run/checkpoints/NeuroLM-B/ckpt.pt'
torch.set_num_threads(4)
with checkpoint.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
record=json.loads((ROOT/'monitor/final_checkpoint.json').read_text());assert sha==record['sha256']
saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
assert saved['epoch']==19 and saved['iter_num']==19140 and saved['epoch_complete']
assert saved['tokenizer_shape']==[8192,128]
state=saved['model']
assert all(torch.isfinite(value).all() for value in state.values() if value.is_floating_point())

def state_digest(prefix):
    digest=hashlib.sha256()
    items=sorted((key[len(prefix):],value) for key,value in state.items() if key.startswith(prefix))
    assert items
    for name,value in items:
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()

done=json.loads((ROOT/'run/completed.json').read_text())
encoder=state_digest('tokenizer.');gpt=state_digest('GPT2.transformer.h.0.')
assert encoder==done['initial']['frozen_encoder_sha256']==done['final']['frozen_encoder_sha256']
assert gpt==done['final']['gpt_block_sha256'] and gpt!=done['initial']['gpt_block_sha256']
metrics=[json.loads(line) for line in (ROOT/'run/metrics.jsonl').read_text().splitlines()]
epochs=[json.loads(line) for line in (ROOT/'run/epochs.jsonl').read_text().splitlines()]
assert [row['step'] for row in metrics]==list(range(1,19141))
assert [row['epoch_completed'] for row in epochs]==list(range(1,21))
assert all(math.isfinite(row[key]) for row in metrics for key in ('eeg_loss','text_loss','grad_norm','lr'))
streams={str(rank):sum(1 for _ in (ROOT/f'run/input_stream_rank-{rank}.jsonl').open()) for rank in range(16)}
assert all(value==19140 for value in streams.values())
start=datetime.datetime.fromisoformat(metrics[0]['utc']);end=datetime.datetime.fromisoformat(done['utc'])
beijing=datetime.timezone(datetime.timedelta(hours=8))
report=dict(ok=True,audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),steps=len(metrics),epochs=len(epochs),
    start_beijing=start.astimezone(beijing).isoformat(),completed_beijing=end.astimezone(beijing).isoformat(),
    elapsed_seconds=(end-start).total_seconds(),first_epoch=epochs[0],last_epoch=epochs[-1],
    last100={key:statistics.mean(row[key] for row in metrics[-100:]) for key in ('eeg_loss','text_loss','eeg_accuracy','text_accuracy')},
    all_metrics_finite=True,all_saved_model_tensors_finite=True,frozen_encoder_unchanged=True,gpt_weights_updated=True,
    complete_input_streams=streams,checkpoint_sha256=sha,checkpoint_bytes=checkpoint.stat().st_size,
    checkpoint_epoch=saved['epoch'],checkpoint_iter=saved['iter_num'],tokenizer_shape=saved['tokenizer_shape'],
    model_args=saved['model_args'],checkpoint_path=str(checkpoint),saved_encoder_sha256=encoder,saved_gpt_block_sha256=gpt)
(ROOT/'monitor/final_audit_20261109.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
