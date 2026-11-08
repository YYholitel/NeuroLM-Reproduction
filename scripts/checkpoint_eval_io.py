"""Small, explicit helpers for trusted project checkpoints and local evidence."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean_state(state):
    cleaned = {}
    for name, tensor in state.items():
        while name.startswith(('_orig_mod.', 'module.')):
            name = name.split('.', 1)[1]
        if name in cleaned:
            raise ValueError('State dictionary prefix normalization collided: ' + name)
        cleaned[name] = tensor
    return cleaned


def tensor_digest(tensor):
    raw = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(raw.shape), str(raw.dtype))).encode())
    digest.update(raw.view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Evidence is immutable: accidental reruns must choose another output path.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
