"""Plot S9."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drought.cli import figure_context

def main():
    ctx=figure_context('S9')
    from drought.enso import plot_enso
    plot_enso(ctx)

if __name__=='__main__':
    main()
