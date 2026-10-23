"""Continental drought-area trends."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drought.cli import figure_context
from drought.time_figures import plot_s8


def main():
    ctx = figure_context("S8")
    plot_s8(ctx)


if __name__ == "__main__":
    main()

