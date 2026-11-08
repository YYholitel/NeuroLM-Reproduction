"""Verify completed five-epoch VQ logs and describe stability without auto-submission."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def verify(root):
    run = root / 'run'
    done = json.loads((run / 'completed.json').read_text())
    assert done['epochs_completed'] == 5 and done['steps'] == 4485
    assert done['partial_epoch_updates'] == 0 and done['world_size'] == 16
    assert done['decoder_layers'] == 3 and done['wandb'] is False
    for rank in [0, 1]:
        assert (root / f'node-{rank}/exit_code.txt').read_text().strip() == '0'
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    epochs = [json.loads(line) for line in (run / 'epoch_metrics.jsonl').read_text().splitlines()]
    assert len(rows) == 4485 and [r['iter'] for r in rows] == list(range(4485))
    assert len(epochs) == 5 and all(e['epoch_complete'] for e in epochs)
    keys = ['reconstruction', 'codebook', 'adversarial', 'total_loss']
    assert all(math.isfinite(r[k]) for r in rows for k in keys)
    config = json.loads((run / 'run_config.json').read_text())
    assert config['epochs'] == 50 and config['warmup_epochs'] == 5
    assert config['max_updates'] == 4485 and config['usage_window'] == 50
    assert config['global_eeg_batch'] == 512 and config['global_text_batch'] == 128
    assert config['vq_codebook_size']==1024 and config['vq_embed_dim']==64
    assert config['vq_dead_code_patience']==50 and config['vq_dead_code_interval']==50
    summaries = []
    for epoch in range(5):
        selected = rows[epoch * 897:(epoch + 1) * 897]
        assert all(r['epoch'] == epoch for r in selected)
        assert epochs[epoch]['epoch'] == epoch + 1
        tail = selected[-100:]
        per_batch = [r['code_usage'] for r in tail]
        percentile = lambda values, q: sorted(values)[round((len(values) - 1) * q)]
        summaries.append(dict(epoch=epoch + 1,
                              mean={k:statistics.mean(r[k] for r in selected) for k in keys},
                              last_100_mean={k:statistics.mean(r[k] for r in tail) for k in keys},
                              last_100_usage=epochs[epoch]['tail_code_usage'],
                              last_100_per_batch={k:dict(min=min(r[k] for r in per_batch),
                                                        p10=percentile([r[k] for r in per_batch], .1),
                                                        median=statistics.median(r[k] for r in per_batch),
                                                        max=max(r[k] for r in per_batch))
                                                  for k in ['active_codes', 'perplexity', 'largest_code_fraction']},
                              resets=sum(r['dead_code_resets'] for r in selected),
                              lr_last=selected[-1]['lr'], alpha_last=selected[-1]['alpha'],
                              fixed_diagnostic=epochs[epoch]['fixed_diagnostic'],
                              review_flags=epochs[epoch]['review_flags']))
    streams = {}
    for rank in range(16):
        data = (run / f'input_stream_rank-{rank}.jsonl').read_bytes()
        assert len(data.splitlines()) == 4485
        streams[str(rank)] = hashlib.sha256(data).hexdigest()
    baseline=json.loads((root.parent/'vq-fixed-e5-20261108-r1/verified.json').read_text())
    assert streams==baseline['input_stream_sha256'],'Input streams differ from baseline'
    result = dict(input_streams_match_baseline=True, verified_complete=True, task_id=json.loads((root/'submitted.json').read_text())['id'],
                  completion=done, epochs=summaries, input_stream_sha256=streams,
                  diagnostic_notice='Training diagnostics only; no claim of held-out quality.',
                  next_stage_submitted=False)
    (root / 'verified.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    verify(parser.parse_args().root)
