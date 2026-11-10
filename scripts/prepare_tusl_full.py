"""Recover all TUSL events without dropping LE or shorter channel montages."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import re

import mne
import numpy as np

CHANNELS = 'FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 A1 A2 FZ CZ PZ T1 T2'.split()
LABELS = {'bckg': 0, 'seiz': 1, 'slow': 2}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select_channels(labels):
    chosen = {}
    for original in labels:
        match = re.fullmatch(r'EEG ([A-Za-z0-9]+)-(REF|LE)', original)
        if match and match[1].upper() in CHANNELS:
            name = match[1].upper()
            if name in chosen:
                raise ValueError(f'Duplicate electrode: {name}')
            chosen[name] = original
    names = [name for name in CHANNELS if name in chosen]
    if not names:
        raise ValueError('No supported EEG channels; refusing to silently skip record')
    originals = [chosen[name] for name in names]
    references = sorted({name.rsplit('-', 1)[1] for name in originals})
    if len(references) != 1:
        raise ValueError('Mixed references require explicit handling')
    return names, originals, references[0]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--existing', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists(), 'Use a new output directory'
    files = sorted(a.raw.rglob('*.edf'))
    assert len(files) == 112
    for split in ('train', 'eval', 'test'):
        (a.out / split).mkdir(parents=True)
    rows = []
    counts = Counter(); labels = Counter(); shapes = Counter(); reused = 0
    for i, path in enumerate(files):
        split = 'train' if i < int(len(files) * .6) else ('test' if i < int(len(files) * .8) else 'eval')
        annotation = path.with_suffix('.tse_agg')
        if not annotation.exists():
            annotation = path.with_suffix('.tse_ag')
        events = [line.split() for line in annotation.read_text().splitlines()[2:] if line.strip()]
        assert events
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose='ERROR')
        names, originals, reference = select_channels(raw.ch_names)
        row = dict(source=str(path.relative_to(a.raw)), source_sha256=sha(path),
                   annotation_sha256=sha(annotation), split=split, ch_names=names,
                   original_ch_names=originals, reference=reference,
                   excluded_outside_author_23_electrodes=[x for x in raw.ch_names if x not in originals],
                   source_sfreq=float(raw.info['sfreq']), events=[])
        expected_paths = [a.existing / split / f'{path.stem}-{j}.pkl' for j in range(len(events))]
        reuse = reference == 'REF' and all(x.exists() for x in expected_paths)
        if not reuse:
            raw.pick(originals).load_data()
            raw.filter(.1, 75., verbose='ERROR')
            raw.notch_filter(60., verbose='ERROR')
            raw.resample(200, n_jobs=1, verbose='ERROR')
            signal = raw.get_data(units='uV'); times = raw.times
        for j, event in enumerate(events):
            start, end = map(float, event[:2]); label = LABELS[event[2]]
            assert abs(end - start - 10.) < 1e-7
            if reuse:
                with expected_paths[j].open('rb') as f:
                    old = pickle.load(f)
                x = old['X']
                assert list(old['ch_names']) == names and int(old['y']) == label
                reused += 1
            else:
                lo, hi = np.searchsorted(times, [start, end], side='left')
                x = signal[:, lo:hi].copy()
            assert x.shape == (len(names), 2000) and np.isfinite(x).all()
            sample = dict(X=x, y=np.array([float(label)]), ch_names=names,
                          original_ch_names=originals, reference=reference,
                          sampling_rate=200, units='uV', source=row['source'],
                          event_seconds=[start, end], preprocessing='0.1-75Hz; notch60Hz; resample200Hz; no /100')
            output = a.out / split / f'{path.stem}-{j}.pkl'
            assert not output.exists(), 'Duplicate output filename'
            with output.open('wb') as f:
                pickle.dump(sample, f, protocol=4)
            if reuse:
                with output.open('rb') as f:
                    check = pickle.load(f)
                assert np.array_equal(old['X'], check['X'])
            row['events'].append(dict(file=str(output.relative_to(a.out)), sha256=sha(output),
                                      label=label, shape=list(x.shape), min=float(x.min()),
                                      max=float(x.max()), mean=float(x.mean()), reused_existing_signal=reuse))
            counts[split] += 1; labels[event[2]] += 1; shapes[str(x.shape)] += 1
        raw.close(); rows.append(row)
        print(json.dumps(dict(record=i+1, split=split, reference=reference, channels=len(names),
                              events=len(events), reused=reuse)), flush=True)
    assert counts == {'train': 174, 'eval': 66, 'test': 60}
    assert labels == {'bckg': 100, 'seiz': 100, 'slow': 100} and reused == 245
    report = dict(ok=True, raw_records=len(files), processed_records=len(rows), skipped_records=0,
                  counts=dict(counts), labels=dict(labels), shapes=dict(shapes), reused_exact_windows=reused,
                  recovered_windows=sum(counts.values())-reused, split='original sorted-file 60/20/20; not patient-independent',
                  mne_version=mne.__version__, numpy_version=np.__version__, records=rows)
    (a.out / 'manifest.json').write_text(json.dumps(report, indent=2))
    print('TUSL_FULL_COMPLETE=' + json.dumps({k:v for k,v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
