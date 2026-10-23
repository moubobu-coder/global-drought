"""Figure S5: ocean and land pressure-level relative humidity."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from drought.cli import figure_context
    ctx = figure_context("S5")
    from drought.style import setup_style
    setup_style()
    from drought.map_figures import render_figS05
    render_figS05(ctx)


if __name__ == "__main__":
    main()
