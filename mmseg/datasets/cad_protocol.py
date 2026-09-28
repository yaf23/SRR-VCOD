"""Shared loader for the CAD frame protocol used in the paper."""

from pathlib import Path, PurePosixPath


CAD_PAPER_FRAME_COUNT = 181
CAD_PAPER_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / 'configs' / 'datasets' / 'cad_paper_frames.txt')


def load_cad_paper_protocol(path=CAD_PAPER_PROTOCOL_PATH):
    """Load and validate the ordered 181-frame CAD paper protocol."""
    protocol_path = Path(path)
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f'CAD frame protocol does not exist: {protocol_path}')

    entries = []
    seen = set()
    with protocol_path.open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            value = line.strip()
            if not value or value.startswith('#'):
                continue
            entry = PurePosixPath(value)
            if (entry.is_absolute() or len(entry.parts) != 2
                    or any(part in ('', '.', '..') for part in entry.parts)
                    or entry.suffix.lower() != '.png'):
                raise ValueError(
                    f'Invalid CAD protocol entry on line {line_number}: '
                    f'{value!r}')
            key = tuple(part.lower() for part in entry.parts)
            if key in seen:
                raise ValueError(
                    f'Duplicate CAD protocol entry on line {line_number}: '
                    f'{value!r}')
            seen.add(key)
            entries.append(entry)

    if len(entries) != CAD_PAPER_FRAME_COUNT:
        raise ValueError(
            f'CAD protocol must contain {CAD_PAPER_FRAME_COUNT} frames; '
            f'found {len(entries)}')
    return entries


def map_processed_cad_frames(image_dir, entries):
    """Map sequential processed CAD frames to paper protocol entries."""
    image_dir = Path(image_dir)
    images = {}
    for path in image_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != '.jpg':
            continue
        if path.stem in images:
            raise ValueError(
                f'Duplicate CAD RGB frame stem in {image_dir}: {path.stem}')
        images[path.stem] = path

    expected = [f'{index:05d}' for index in range(len(entries))]
    missing = sorted(set(expected) - images.keys())
    extra = sorted(images.keys() - set(expected))
    if missing or extra:
        raise ValueError(
            f'CAD processed frames in {image_dir} must be numbered '
            f'00000..{len(entries) - 1:05d}; '
            f'missing={missing[:10]}, extra={extra[:10]}')
    return list(zip(entries, (images[stem] for stem in expected)))


__all__ = [
    'CAD_PAPER_FRAME_COUNT', 'CAD_PAPER_PROTOCOL_PATH',
    'load_cad_paper_protocol', 'map_processed_cad_frames'
]
