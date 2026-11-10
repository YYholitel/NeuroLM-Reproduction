"""Assemble isolated official-init rerun inputs from previously verified artifacts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import tarfile


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def verified_copy(src, dst, manifest):
    for name, expected in manifest.items():
        assert sha(src / name) == expected, name
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reuse-validated-assets', action='store_true')
    a = p.parse_args(); root = a.root; base = root.parent; repo = Path(__file__).resolve().parents[1]
    tusl = root / 'data/TUSL'; report = json.loads((tusl / 'manifest.json').read_text())
    assert report['ok'] and report['processed_records'] == 112 and report['skipped_records'] == 0
    for row in report['records']:
        for event in row['events']:
            assert sha(tusl / event['file']) == event['sha256']
    if a.reuse_validated_assets:
        assets = root / 'assets'
        ready = json.loads((assets / 'data_ready.json').read_text())
        assert ready['ready'] and ready['total_samples'] == 459711
        assert sha(assets/'train_files.txt') == ready['train_manifest_sha256']
        assert sha(tusl/'manifest.json') == ready['tusl_manifest_sha256']
        counts = ready['datasets']
    else:
        for name in ['TUAB', 'TUEV', 'HMC', 'EEGWorkload', 'text']:
            (root / 'data' / name).symlink_to(base / 'data' / name, target_is_directory=True)
        old = base / 'pretrain-full20-20261109-r1/assets'
        assets = root / 'assets'; assets.mkdir(); (assets / 'shards').mkdir()
        (assets / 'text').symlink_to(old / 'text', target_is_directory=True)
        rows = []
        for line in (old / 'train_files.txt').read_text().splitlines():
            row = json.loads(line)
            assert row['name'].startswith('train/')
            if row['name'].split('/')[1] != 'TUSL':
                row['shard'] = str(old / row['shard']); rows.append(row)
        shard = assets / 'shards/TUSL-full.tar'
        with tarfile.open(shard, 'w') as tar:
            for path in sorted((tusl / 'train').glob('*.pkl')):
                tar.add(path, arcname='train/TUSL/' + path.name, recursive=False)
        with tarfile.open(shard) as tar:
            for member in tar:
                if member.isfile():
                    path = tusl / 'train' / Path(member.name).name
                    rows.append(dict(name=member.name, shard=str(shard), offset=member.offset_data,
                                     size=member.size, sha256=sha(path)))
        counts = Counter(row['name'].split('/')[1] for row in rows)
        assert len(rows) == 459711 and len({r['name'] for r in rows}) == len(rows)
        assert counts == {'TUAB': 298511, 'TUEV': 68270, 'HMC': 91248, 'Workload': 1508, 'TUSL': 174}
        manifest = assets / 'train_files.txt'
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        dump(assets / 'data_ready.json', dict(ready=True, diagnostic=False, total_samples=len(rows),
            datasets=dict(counts), train_manifest_sha256=sha(manifest), tusl_manifest_sha256=sha(tusl/'manifest.json'),
            reused_non_tusl_source=str(old), original_source_plan_sha256=sha(old/'source_plan.json'),
            split='train only; original validation/test excluded', tusl_records=112, tusl_skipped_records=0))
    code = root / 'code'; author = root / 'author'
    s2_manifest = json.loads((repo/'results/pretrain_full20_20261109/final/code/bundle_manifest.json').read_text())
    verified_copy(base/'pretrain-full20-20261109-r1/code', code, s2_manifest)
    shutil.copyfile(repo/'scripts/pretrain_official_runtime.py', code/'pretrain_minimal.py')
    shutil.copyfile(repo/'scripts/official_checkpoint_init.py', code/'official_checkpoint_init.py')
    path = code/'train_pretrain.py'; text = path.read_text()
    text = replace_once(text, "    parser.add_argument('--max_updates', default=0, type=int)",
        "    parser.add_argument('--max_updates', default=0, type=int)\n    parser.add_argument('--initial_neurolm_path', type=str, required=True)")
    path.write_text(text)
    s3_manifest = json.loads((repo/'results/stage3_new_pretrain_20261109/final/author/bundle_manifest.json').read_text())
    verified_copy(base/'stage3-new-pretrain-20261109-r1/author', author, s3_manifest)
    path = author/'downstream_dataset.py'; text = path.read_text(); i = text.index('class TUSLLoader(')
    text = text[:i] + replace_once(text[i:], '        ch_names = self.ch_names',
        "        ch_names = list(sample['ch_names'])\n        assert len(ch_names) == X.shape[0] and len(set(ch_names)) == len(ch_names)\n        assert all(name in self.ch_names for name in ch_names)")
    path.write_text(text)
    path = author/'train_instruction.py'; text = path.read_text()
    needle = 'rank=ddp_rank, shuffle=True'
    assert text.count(needle) == 2
    # Prevent DistributedSampler's ceil-padding from adding an extra microstep
    # beyond the author's floor-based schedule at this dataset size.
    text = text.replace(needle, 'rank=ddp_rank, shuffle=True, seed=args.seed, drop_last=True')
    path.write_text(text)
    for folder, original in [(code, s2_manifest), (author, s3_manifest)]:
        current = {name:sha(folder/name) for name in original}
        if folder == code:current['official_checkpoint_init.py'] = sha(folder/'official_checkpoint_init.py')
        assert all(current[n] == h for n,h in original.items() if n.startswith('model/'))
        dump(folder/'bundle_manifest.json', current)
    old_spec = json.loads((repo/'results/pretrain_full20_20261109/final/task.json').read_text())
    for seed in (42, 43, 44):
        for stage in (2, 3):
            job = root/f'seed-{seed}/stage{stage}'; job.mkdir(parents=True)
            (job/'assets').symlink_to(assets, target_is_directory=True)
            spec = json.loads(json.dumps(old_spec))
            spec['name'] = f'neurolm-official-tusl-s{seed}-stage{stage}-20261111-r1'
            if stage == 3:spec['resource'].update(cpu=28, memory=224, gpu=2, instance_num=1)
            spec['cos_output'] = 'omni-eeg-1442740494/lhy/neurolm-reproduction/outputs/'+spec['name']+'/'
            spec['command'] = f'bash {root}/scripts/run_official_stage{stage}.sh {job} {seed}'
            dump(job/'task.json', spec)
            control = (repo/'scripts/pretrain_full20_control.py').read_text()
            control = replace_once(control, "ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-full20-20261109-r1')", f'ROOT=Path({str(job)!r})')
            control = control.replace('459656', '459711')
            (job/'control.py').write_text(control)
            notebook = (repo/'scripts/pretrain_full20_notebook.py').read_text()
            notebook = notebook.replace('/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-full20-20261109-r1/control.py', str(job/'control.py'))
            (job/'notebook.py').write_text(notebook)
    dump(root/'build.json', dict(ok=True, seeds=[42,43,44], train_counts=dict(counts),
         models_unchanged=True, official_checkpoint_initialization=True, stage2_epochs=20, stage3_epochs=5))
    print(json.dumps(dict(ok=True, counts=dict(counts), root=str(root))), flush=True)


if __name__ == '__main__':main()
