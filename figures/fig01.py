"""Global annual meteorological drivers."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drought.cli import figure_context
from drought.time_figures import plot_fig1


def main():
    ctx = figure_context("Fig1")
    plot_fig1(ctx)


if __name__ == "__main__":
    main()

