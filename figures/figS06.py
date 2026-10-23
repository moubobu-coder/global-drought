"""Continental December SPEI-12 series."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drought.cli import figure_context
from drought.time_figures import plot_s6


def main():
    ctx = figure_context("S6")
    plot_s6(ctx)


if __name__ == "__main__":
    main()

