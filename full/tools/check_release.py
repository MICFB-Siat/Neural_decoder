
import argparse
import ast
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'validation/structure_audit.json')
    args = p.parse_args()
    registry = json.loads((ROOT / 'workflows.json').read_text())
    assets = json.loads((ROOT / 'asset_manifest.json').read_text())['files']
    resources = {str(ROOT / item['path']) for item in assets}
    resource_dirs = {str(parent) for path in resources for parent in Path(path).parents}
    broken, external, syntax = [], [], []
    for item in registry.values():
        path = ROOT / item['directory'] / item['entry']
        if not path.is_file():
            broken.append(str(path.relative_to(ROOT)))
    checked = 0
    for parent, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {'.cache', '__pycache__', 'outputs', 'output', 'validation'}]
        for name in dirs + files:
            path = Path(parent) / name
            if path.is_symlink():
                target = path.resolve()
                if not target.is_relative_to(ROOT):
                    external.append(str(path.relative_to(ROOT)))
                elif not target.exists() and str(target) not in resources | resource_dirs:
                    broken.append(str(path.relative_to(ROOT)))
            elif path.suffix == '.py':
                checked += 1
                try:
                    ast.parse(path.read_bytes(), filename=str(path))
                except SyntaxError as error:
                    syntax.append({'path': str(path.relative_to(ROOT)), 'error': str(error)})
    result = dict(workflows=len(registry), python_files_checked=checked, broken_links=broken,
                  external_links=external, syntax_errors=syntax)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(broken or external or syntax))


if __name__ == '__main__':
    main()
