"""Plot Figure S11: regional annual precipitation comparisons."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    from drought.cli import figure_context
    ctx=figure_context('S11')
    from drought.regional_precipitation import plot_regional_precipitation
    plot_regional_precipitation(ctx)

if __name__=='__main__':
    main()
