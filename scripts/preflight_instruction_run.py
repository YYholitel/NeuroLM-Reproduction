"""Fail before GPU allocation if the verified five-task inputs are incomplete."""
import argparse
import hashlib
import json
from pathlib import Path

SPLITS = {
    'TUAB/processed/train': 298511, 'TUAB/processed/val': 73627, 'TUAB/processed/test': 36945,
    'TUEV/processed_train': 68270, 'TUEV/processed_eval': 15662, 'TUEV/processed_test': 28305,
    'HMC/train': 91248, 'HMC/eval': 22124, 'HMC/test': 23871,
    'EEGWorkload/train': 1508, 'EEGWorkload/eval': 290, 'EEGWorkload/test': 290,
    'TUSL/train': 119, 'TUSL/eval': 66, 'TUSL/test': 60,
}
HASHES = {
    'NeuroLM-B.pt': 'ffe098bc138b89f8817d3710a3604498d8ecd15135080e2ca27735d05c6d29ab',
    'VQ.pt': 'e792c39a6a9e6d1bf4604cf63090730424f1d37f942597883d0c0a1375a2663a',
    'train.bin': '68a352b5dbd52e85b7a8a1556c1ccae65ce14b02e4913788b0883b89ceddc1d1',
    'val.bin': '009ecdd6da34f1ae915ff1f0f3aeb0416b02f2252356505b32ce44166dd5cfcc',
}


def check(data_dir, checkpoint_dir):
    report = {'counts': {}, 'sha256': {}, 'errors': [], 'paper_comparable': False}
    for relative, expected in SPLITS.items():
        folder = data_dir / relative
        actual = sum(p.is_file() and p.suffix == '.pkl' for p in folder.iterdir()) if folder.is_dir() else 0
        report['counts'][relative] = {'actual': actual, 'expected': expected}
        if actual != expected:
            report['errors'].append(f'{relative}: expected {expected}, found {actual}')
    for name, expected in HASHES.items():
        path = (checkpoint_dir if name.endswith('.pt') else data_dir / 'text') / name
        if not path.is_file():
            report['errors'].append(f'Missing {name}')
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        report['sha256'][name] = digest
        if digest != expected:
            report['errors'].append(f'Checksum mismatch: {name}')
    report['passed'] = not report['errors']
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.data_dir, args.checkpoint_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
