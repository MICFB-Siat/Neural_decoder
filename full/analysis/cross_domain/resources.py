
from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p/'checkpoint_manifest.json').is_file())
_release_sys.path.insert(0, str(_release_root))
from release_checkpoints import resolve_weight as _weight_path

from pathlib import Path
import json, os
ROOT=Path(__file__).resolve().parent

def resolve(value):
    if value is None: return None
    config=Path(os.environ.get('FIGURE_RESOURCES',ROOT/'resources.json'))
    spec=json.loads(config.read_text());key=str(value)
    if key in spec['local']:path=ROOT/spec['local'][key]
    elif key in spec['external']:
        entry=spec['external'][key]
        if entry['kind'] not in ('dataset','weight'):raise ValueError('External non-dataset resource is forbidden')
        path=Path(entry['path'])
    else:
        path=Path(value)
        if not path.is_absolute():path=ROOT/path
        if not path.resolve().is_relative_to(ROOT):raise ValueError('Unregistered external resource: '+str(value))
    path=_weight_path(path)
    if not path.exists():raise FileNotFoundError(path)
    return path
