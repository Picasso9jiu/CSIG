#!/usr/bin/env python3
"""Package the DREAM code-review materials without including the dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


# Never put local datasets or generated/runtime files into the review archive.
# The release repository normally has no NPZ data, but these guards also keep
# the package safe if an evaluator places the official data beside the code.
EXCLUDE_DIRS = {
    '.git', 'outputs', 'log', 'runs', 'work', 'review_packages',
    '__pycache__', '.pytest_cache',
    '训练集、验证集', '测试集', 'EV-UAV-dataset',
}
EXCLUDE_SUFFIXES = {'.pyc', '.pyo', '.so', '.npz'}
REQUIRED = [
    'README.md',
    'requirements.txt',
    'requirements-cuda111.txt',
    'environment.yml',
    'NOTICE.md',
    'configs',
    'dataset',
    'model',
    'utils',
    'lib/hais_ops',
    'scripts',
    'checkpoints',
    'docs/ALGORITHM_REPORT_DREAM.md',
    'docs/RUNNING_GUIDE.md',
    'docs/REVIEW_CHECKLIST.md',
    'note.md',
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    try:
        data = path.read_bytes()[:200]
    except OSError:
        return False
    return data.startswith(b'version https://git-lfs.github.com/spec/v1')


def iter_files(root: Path):
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        yield path, relative


def check_required(root: Path) -> None:
    missing = []
    for item in REQUIRED:
        path = root / item
        if not path.exists():
            missing.append(item)
    if missing:
        raise FileNotFoundError('missing required release material(s): ' + ', '.join(missing))


def clean_name(value: str, field: str) -> str:
    """Validate a human-provided naming component before using it as a path."""
    value = str(value).strip()
    if not value:
        raise ValueError('{} must not be empty'.format(field))
    if value in {'.', '..'} or '/' in value or '\\' in value:
        raise ValueError('{} must be a single filename component'.format(field))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', required=True, help='official platform ranking number')
    parser.add_argument('--team-cn', required=True, help='official Chinese team name')
    parser.add_argument('--platform-en', required=True, help='official platform English name')
    parser.add_argument('--track', required=True, help='track number, for example 1')
    parser.add_argument('--output-dir', type=Path, default=Path('review_packages'))
    parser.add_argument('--without-artifacts', action='store_true',
                        help='omit final M244 submission/audit artifacts (not recommended)')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    check_required(root)
    rank = clean_name(args.rank, 'rank')
    team_cn = clean_name(args.team_cn, 'team-cn')
    platform_en = clean_name(args.platform_en, 'platform-en')
    track = clean_name(args.track, 'track')

    archive_name = '{}-{}-{}-赛道{}.zip'.format(
        rank,
        team_cn,
        platform_en,
        track,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / archive_name
    if output.exists():
        raise FileExistsError('refusing to overwrite existing archive: {}'.format(output))

    selected = []
    for source, relative in iter_files(root):
        relative_text = relative.as_posix()
        if args.without_artifacts and relative_text.startswith('artifacts/'):
            continue
        selected.append((source, relative))
    if not args.without_artifacts:
        selected_artifacts = [p for p in selected if p[1].as_posix().startswith('artifacts/')]
        if not selected_artifacts:
            raise FileNotFoundError('include-artifacts requested, but artifacts/ is empty')

    pointer_files = [str(path.relative_to(root)) for path, _ in selected if is_lfs_pointer(path)]
    if pointer_files:
        raise RuntimeError(
            'Git LFS pointer files detected; run `git lfs pull` before packaging: '
            + ', '.join(pointer_files[:10])
        )

    manifest = {
        'schema': 'evsod-review-materials-v1',
        'release': 'DREAM/M244',
        'archive_name': archive_name,
        'dataset_included': False,
        'artifacts_included': not bool(args.without_artifacts),
        'file_count': len(selected),
        'files': [],
    }
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, relative in selected:
            archive.write(source, arcname='EVSOD/' + relative.as_posix())
            manifest['files'].append({
                'path': relative.as_posix(),
                'size': source.stat().st_size,
                'sha256': sha256(source),
            })
        # Use a real trailing newline so the manifest is a normal UTF-8 JSON
        # text file when extracted by the review team.
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        archive.writestr('EVSOD/review_materials_manifest.json', manifest_bytes)

    print('archive:', output)
    print('files:', len(selected))
    print('dataset_included: false')
    print('artifacts_included:', not bool(args.without_artifacts))
    print('sha256:', sha256(output))


if __name__ == '__main__':
    main()
