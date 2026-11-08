"""Apply patch 0017 to a new runtime directory after verifying its exact baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-runtime', type=Path, required=True)
    parser.add_argument('--output-runtime', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_runtime.resolve()
    output = args.output_runtime.resolve()
    if output.exists() or source == output or source in output.parents:
        raise ValueError('Output must be a new directory outside the source runtime')
    repo = Path(__file__).resolve().parents[1]
    checks_path = repo/'results/vq_codebook_fix_checks_20261108.json'
    checks = json.loads(checks_path.read_text(encoding='utf-8'))
    patch = repo/'patches/0017-vq-valid-token-ema-and-dead-code-reset.patch'
    assert sha(patch) == checks['patch_sha256'], 'Patch checksum mismatch'
    baseline = json.loads((source/'bundle_manifest.json').read_text())
    for name, digest in baseline.items():
        assert sha(source/name) == digest, ('Baseline runtime hash mismatch', name)
    for name, digest in checks['changed_files'].items():
        assert sha(source/name) in (digest['before'], digest['normalized_before']), name
    with tempfile.TemporaryDirectory(prefix='neurolm-vq-patch-') as directory:
        staged = Path(directory)
        for name in checks['changed_files']:
            (staged/name).parent.mkdir(parents=True, exist_ok=True)
            (staged/name).write_bytes((source/name).read_text(encoding='utf-8').encode('utf-8'))
        # Prevent an unrelated parent repository from filtering patch paths.
        env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(staged.parent))
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'apply', '--check', str(patch)],
                       cwd=staged, env=env, check=True)
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'apply', str(patch)],
                       cwd=staged, env=env, check=True)
        for name, digest in checks['changed_files'].items():
            assert sha(staged/name) == digest['after'], ('Patched hash mismatch', name)
        shutil.copytree(source, output, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
        for name in checks['changed_files']:
            shutil.copyfile(staged/name, output/name)
    manifest = {name: sha(output/name) for name in baseline}
    (output/'bundle_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(ok=True, output_runtime=str(output), source_unchanged=True,
                         patch_sha256=checks['patch_sha256'])))


if __name__ == '__main__':
    main()
