"""Global annual drought-area fractions."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drought.cli import figure_context
from drought.time_figures import plot_fig3


def main():
    ctx = figure_context("Fig3")
    plot_fig3(ctx)


if __name__ == "__main__":
    main()

