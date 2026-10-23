"""Figure S7: SPEI-AEDclim trends and paired-difference trends."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from drought.cli import figure_context
    ctx = figure_context("S7")
    from drought.style import setup_style
    setup_style()
    from drought.map_figures import render_figS07
    render_figS07(ctx)


if __name__ == "__main__":
    main()
