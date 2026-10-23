"""Figure 2: SPEI-12 spatial trends and global annual series."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from drought.cli import figure_context
    ctx = figure_context("Fig2")
    from drought.style import setup_style
    setup_style()
    from drought.map_figures import render_fig02
    render_fig02(ctx)


if __name__ == "__main__":
    main()
