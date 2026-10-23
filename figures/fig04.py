"""Plot Fig4."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drought.cli import figure_context

def main():
    ctx=figure_context('Fig4')
    from drought.annual_summary import plot_annual_summary
    plot_annual_summary(ctx)

if __name__=='__main__':
    main()
