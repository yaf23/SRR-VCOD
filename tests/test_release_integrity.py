from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_files_exist():
    required = {
        'README.md',
        'README.zh-CN.md',
        'LICENSE',
        'LICENSES/Apache-2.0.txt',
        'docs/INSTALL.md',
        'docs/DATASETS.md',
        'docs/CHECKPOINTS.md',
        'docs/CONFIGS.md',
    }
    assert all((REPO_ROOT / path).is_file() for path in required)


def test_cad_paper_protocol_contains_181_unique_frames():
    manifest = REPO_ROOT / 'configs/datasets/cad_paper_frames.txt'
    entries = [
        line.strip()
        for line in manifest.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]
    assert len(entries) == 181
    assert len(set(entries)) == 181
