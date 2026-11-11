import collections
import datetime
import hashlib
import json
from pathlib import Path
import re

import mne

raw_root = Path('/data/lihy/NeuroLM/datasets/raw/tuh/tusl/v2.0.1')
data_root = Path('/data/lihy/NeuroLM/datasets/TUSL-full-20261111-r1')
output = Path('/data/lihy/NeuroLM/outputs/data-audit-20261111-1714')
output.mkdir(parents=True, exist_ok=True)
manifest = json.loads((data_root/'manifest.json').read_text())
records = {r['source']: r for r in manifest['records']}
allowed = 'FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 A1 A2 FZ CZ PZ T1 T2'.split()
labels = {'bckg':0, 'seiz':1, 'slow':2}
gold, errors, groups = [], [], collections.Counter()
paths = sorted(raw_root.rglob('*.edf'))
for i, path in enumerate(paths):
    source = str(path.relative_to(raw_root))
    split = 'train' if i < int(len(paths)*.6) else ('test' if i < int(len(paths)*.8) else 'eval')
    record = records.get(source)
    if record is None:
        errors.append(dict(source=source, error='raw EDF missing from preprocessing manifest'))
        continue
    annotation = path.with_suffix('.tse_agg')
    if not annotation.exists(): annotation = path.with_suffix('.tse_ag')
    annotation_hash = hashlib.sha256(annotation.read_bytes()).hexdigest()
    if annotation_hash != record['annotation_sha256']:
        errors.append(dict(source=source, error='annotation hash mismatch'))
    lines = [line.split() for line in annotation.read_text().splitlines()
             if line.strip() and not line.strip().startswith(('#', 'version'))]
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose='ERROR')
    electrodes = {}
    for original in raw.ch_names:
        match = re.fullmatch(r'EEG ([A-Za-z0-9]+)-(REF|LE)', original)
        if match and match[1].upper() in allowed:
            assert match[1].upper() not in electrodes
            electrodes[match[1].upper()] = original
    channels = [name for name in allowed if name in electrodes]
    originals = [electrodes[name] for name in channels]
    reference = sorted({s.rsplit('-',1)[1] for s in originals})
    if record['split'] != split or record['ch_names'] != channels or record['original_ch_names'] != originals:
        errors.append(dict(source=source, error='split/channel manifest mismatch'))
    if len(lines) != len(record['events']):
        errors.append(dict(source=source, error='event count mismatch'))
    for j, line in enumerate(lines):
        start, end = map(float,line[:2]); label=labels[line[2]]
        expected=f'{split}/{path.stem}-{j}.pkl'
        row=dict(file=expected, source=source, label=label, raw_label=line[2],
            start=start,end=end,split=split,channels=channels,original_ch_names=originals,
            reference=reference[0],source_sfreq=float(raw.info['sfreq']))
        gold.append(row);groups[(split,len(channels),reference[0],label)]+=1
        if end-start != 10 or j>=len(record['events']) or record['events'][j]['file']!=expected or record['events'][j]['label']!=label:
            errors.append(dict(file=expected,error='annotation/manifest window or label mismatch'))
    raw.close()
report=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    raw_records=len(paths),annotations=len(paths),windows=len(gold),errors=errors,
    groups=[dict(split=k[0],channels=k[1],reference=k[2],label=k[3],windows=v) for k,v in sorted(groups.items())],
    records=gold,source='independent raw annotations and EDF headers',
    manifest_sha256=hashlib.sha256((data_root/'manifest.json').read_bytes()).hexdigest(),
    full_edf_content_hashes_recomputed=False)
(output/'tusl_raw_gold.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)
