"""Freeze the external Metal sources used by one benchmark executable."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def snapshot_shaders(output):
    dest = Path(output).resolve() / 'metal'
    dest.mkdir()
    overrides, manifest = {}, {}
    for source in sorted(Path('metal').glob('*.metal')):
        name = f'DS4_METAL_{source.stem.upper()}_SOURCE'
        target = dest / source.name
        source = Path(os.environ.get(name) or source)
        shutil.copy2(source, target)
        overrides[name] = str(target)
        manifest[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (dest/'sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return overrides
