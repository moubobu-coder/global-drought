"""Continental temperature, radiation and wind."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drought.cli import figure_context
from drought.time_figures import plot_continental_forcing


def main():
    ctx = figure_context("S2")
    plot_continental_forcing(ctx, "S2")


if __name__ == "__main__":
    main()

