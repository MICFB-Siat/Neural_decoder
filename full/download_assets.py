

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', action='append', help='Release-relative directory; repeatable. Use all for every asset.')
    p.add_argument('--list', action='store_true', help='List resource groups and sizes')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verify', action='store_true', help='Check local sizes and SHA-256 without downloading')
    p.add_argument('--local-dir', type=Path, default=ROOT)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    manifest = json.loads((ROOT / 'asset_manifest.json').read_text())
    if args.workers < 1:
        p.error('--workers must be positive')
    if args.list:
        groups = {}
        for item in manifest['files']:
            parts = item['path'].split('/')
            group = '/'.join(parts[:2]) if parts[0] != 'decoding' else 'decoding'
            n, size = groups.get(group, (0, 0))
            groups[group] = n + 1, size + item['size']
        for group, (n, size) in sorted(groups.items()):
            print(f'{group:32s} {n:6d} files {size / 1e9:8.3f} GB')
        return
    if not args.scope:
        p.error('Choose --scope DIRECTORY or --scope all; use --list to see groups')
    selected = {}
    for scope in args.scope:
        scope = scope.strip('/')
        matches = [i for i in manifest['files'] if scope == 'all' or i['path'] == scope or i['path'].startswith(scope + '/')]
        if not matches:
            p.error(f'No assets match {scope!r}; use --list')
        selected.update((i['path'], i) for i in matches)
    files = list(selected.values())
    print(f"Selected {len(files)} files, {sum(i['size'] for i in files) / 1e9:.3f} GB")
    if args.dry_run:
        return
    if args.verify:
        failures = []
        for item in files:
            path = args.local_dir / item['path']
            if not path.is_file() or path.stat().st_size != item['size']:
                failures.append(item['path'] + ': missing or wrong size')
            elif item.get('sha256'):
                with path.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                        failures.append(item['path'] + ': SHA-256 mismatch')
        for failure in failures:
            print(failure)
        print(f'{len(files) - len(failures)}/{len(files)} resources verified')
        raise SystemExit(bool(failures))
    from huggingface_hub import hf_hub_download

    def download(item):
        target = args.local_dir / item['path']
        if target.is_file() and target.stat().st_size == item['size']:
            with target.open('rb') as stream:
                if item.get('sha256') and hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']:
                    return item['path'] + ' (already verified)'
        staging = args.local_dir / '.cache' / 'resource_download'
        source = Path(hf_hub_download(
            repo_id=manifest['repo_id'],
            filename=manifest['remote_prefix'] + '/' + item['path'],
            revision=item.get('revision', manifest['revision']), local_dir=staging,
        ))
        target = args.local_dir / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_size != item['size']:
            raise ValueError(f'Unexpected size for {item["path"]}')
        if item.get('sha256'):
            with source.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                    raise ValueError(f'Checksum mismatch for {item["path"]}')
        source.replace(target)
        return item['path']

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, name in enumerate(pool.map(download, files), 1):
            print(f'[{index}/{len(files)}] {name}', flush=True)


if __name__ == '__main__':
    main()
