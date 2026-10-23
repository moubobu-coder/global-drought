"""Shared figure typography and export."""
from __future__ import annotations


def setup_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'legend.frameon': False,
    })


def save_figure(fig, ctx):
    import matplotlib.pyplot as plt
    ctx.output.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for suffix in ctx.formats:
        kwargs = {'dpi': ctx.dpi, 'bbox_inches': 'tight'}
        if suffix in ('tif', 'tiff'):
            kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(ctx.output / f'{ctx.figure_id}.{suffix}', **kwargs)
    plt.close(fig)
    print(f'{ctx.figure_id}: saved to {ctx.output}')
