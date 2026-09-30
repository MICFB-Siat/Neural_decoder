from pathlib import Path
import json, os
ROOT = Path(__file__).resolve().parent

def path(name):
    values = json.loads((ROOT / 'paths.json').read_text())
    override = os.environ.get('FIGURE_PATHS')
    config = Path(os.path.expandvars(override)).expanduser() if override else ROOT / 'paths.local.json'
    if override or config.is_file():
        values.update(json.loads(config.read_text()))
    if name not in values:
        raise KeyError(f'Missing resource {name} in {config}')
    if not isinstance(values[name], str) or not values[name].strip():
        raise ValueError(f'Resource {name} must be a nonempty path string')
    value = Path(os.path.expandvars(values[name])).expanduser()
    return value if value.is_absolute() else ROOT / value
