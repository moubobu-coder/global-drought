"""Common command-line options for manuscript figures."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FigureContext:
    figure_id: str
    root: Path
    output: Path
    config: dict
    formats: tuple[str, ...]
    dpi: int
    check_inputs: bool

    @property
    def input_root(self) -> Path:
        return self.root

    def path(self, key: str, default: str | Path) -> Path:
        value = self.config.get('paths', {}).get(key, self.config.get(key, default))
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path


def figure_context(figure_id: str) -> FigureContext:
    raw_id = figure_id
    if figure_id.startswith('FigS'):
        figure_id = figure_id[3:]
    parser = argparse.ArgumentParser(description=f'Plot manuscript figure {figure_id}.')
    parser.add_argument('--input-root', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('figure_output'))
    parser.add_argument('--config', type=Path, help='JSON paths and figure-specific options.')
    parser.add_argument('--formats', default='pdf,svg,png', help='Comma-separated export formats.')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--check-inputs', action='store_true', help='Check inputs without drawing.')
    args = parser.parse_args()
    config = {}
    if args.config:
        raw = json.loads(args.config.read_text(encoding='utf-8-sig'))
        config = {k: v for k, v in raw.items() if k not in ('figures',)}
        figures = raw.get('figures', {})
        specific = figures.get(figure_id, figures.get(raw_id, figures.get(figure_id.lower(), {})))
        config.update({k: v for k, v in specific.items() if k != 'paths'})
        config['paths'] = {**raw.get('paths', {}), **specific.get('paths', {})}
    formats = tuple(s.strip().lower() for s in args.formats.split(',') if s.strip())
    if not formats or any(s not in ('pdf', 'svg', 'png', 'tif', 'tiff', 'eps') for s in formats):
        parser.error('Supported formats: pdf, svg, png, tif, tiff, eps.')
    if args.dpi <= 0:
        parser.error('--dpi must be positive.')
    return FigureContext(figure_id, args.input_root.resolve(), args.output_dir.resolve(),
                         config, formats, args.dpi, args.check_inputs)
