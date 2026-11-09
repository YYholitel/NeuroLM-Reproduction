"""Copy the verified baseline runtime; defer evaluation without changing training math."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser(__doc__)
p.add_argument('--source-author',type=Path,required=True)
p.add_argument('--source-manifest',type=Path,required=True)
p.add_argument('--output-author',type=Path,required=True)
a=p.parse_args()
manifest=json.loads(a.source_manifest.read_text())
source={name.removeprefix('author/'):sha for name,sha in manifest.items() if name.startswith('author/')}
for name,expected in source.items():assert hashlib.sha256((a.source_author/name).read_bytes()).hexdigest()==expected,name
assert not a.output_author.exists()
shutil.copytree(a.source_author,a.output_author,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
path=a.output_author/'train_instruction.py';original=path.read_bytes();text=original.decode()
newline='\r\n' if b'\r\n' in original else '\n'
needle="    parser.add_argument('--eval_only', default=False, action='store_true')"
assert text.count(needle)==1
text=text.replace(needle,needle+newline+"    parser.add_argument('--skip_evaluation', default=False, action='store_true')")
needle='        # validation and test'
assert text.count(needle)==1
text=text.replace(needle,newline.join(["        if args.skip_evaluation:","            print('Evaluation deferred to the audited constrained runner.', flush=True)","            continue","",needle]))
needle="'evaluation': 'greedy_top_k_1_max_5_tokens_author_parser'"
assert text.count(needle)==1
text=text.replace(needle,"'evaluation': 'deferred_constrained' if args.skip_evaluation else 'greedy_top_k_1_max_5_tokens_author_parser'")
path.write_bytes(text.encode())
runtime=dict(source);runtime['train_instruction.py']=hashlib.sha256(path.read_bytes()).hexdigest()
assert all(hashlib.sha256((a.output_author/name).read_bytes()).hexdigest()==sha for name,sha in runtime.items())
(a.output_author/'bundle_manifest.json').write_text(json.dumps(runtime,indent=2))
print(json.dumps(dict(ok=True,changed_files=['train_instruction.py'],model_files_unchanged=True,
                     training_objective_optimizer_and_schedule_unchanged=True,before=source['train_instruction.py'],
                     after=runtime['train_instruction.py'])))
