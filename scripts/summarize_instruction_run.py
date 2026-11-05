"""Validate full evaluation coverage and emit a small, path-free run summary."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {'TUAB': (73627, 36945), 'TUEV': (15662, 28305),
            'HMC': (22124, 23871), 'Workload': (290, 290), 'TUSL': (66, 60)}


def summarize(run):
    completed = json.loads((run / 'completed.json').read_text())
    manifest = json.loads((run / 'run_manifest.json').read_text())
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert completed['epochs'] == 1 and completed['seed'] == 42
    assert len(rows) == 5 and {r['dataset'] for r in rows} == set(EXPECTED)
    for row in rows:
        assert row['epoch'] == 0
        for split, count in zip(('val', 'test'), EXPECTED[row['dataset']]):
            assert row[split]['samples'] == count, (row['dataset'], split)
            files = sorted((run / 'predictions').glob(f"{row['dataset']}-{split}-epoch0-rank*.jsonl"))
            assert len(files) == manifest['world_size']
            indices = set()
            failures = 0
            for file in files:
                with file.open() as stream:
                    for line in stream:
                        item = json.loads(line)
                        assert item['index'] not in indices, 'Duplicate prediction'
                        indices.add(item['index'])
                        failures += item['prediction'] == -1
            assert indices == set(range(count)), 'Missing prediction'
            assert failures == row[split]['parse_failures']
    checkpoint = run / 'checkpoints/instruction-B/ckpt.pt'
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    summary = {
        'status': 'completed', 'category': 'five_task_exploratory_modified_protocol',
        'paper_comparable': False, 'seed': 42, 'epochs': 1,
        'checkpoint_sha256': digest, 'checkpoint_bytes': checkpoint.stat().st_size,
        'metrics': rows,
        'runtime': {k: manifest[k] for k in ('world_size', 'torch', 'cuda', 'dtype', 'gpu',
                    'global_eeg_batch', 'global_text_batch', 'microsteps_per_epoch', 'dataset_counts')},
        'metric_caveat': 'Author generation parser and hard-label AUC retained; parse failures reported explicitly.',
    }
    (run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    print(json.dumps(summarize(parser.parse_args().run_dir), indent=2))
