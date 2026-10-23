"""Annual SSI/SPEI means, quartiles and Sen trends."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .stats import ktaub,source_sen_line
from .style import save_figure

COLORS={'12-month SSI':'#007FAF','GHCD':'#DF5C4B','ERA5':'#4B7888','JRA-3Q':'#EC8A1C'}
DISPLAY={'SSI-12':'12-month SSI','GHCD SPEI-12':'GHCD','ERA5 SPEI-12':'ERA5','JRA-3Q SPEI-12':'JRA-3Q'}

def p_text(value):
    return 'p < 0.001' if value < 0.001 else f'p = {value:.4f}' if value < 0.01 else f'p = {value:.3f}'

def read_annual_summary(path):
    if path.suffix.lower()=='.json':
        raw=json.loads(path.read_text(encoding='utf-8-sig'))
        return {'years':raw['years'],'series':{key:{f:raw['series'][key][f] for f in ('mean','p25','p75')} for key in DISPLAY}}
    table=pd.read_csv(path)
    required={'year','series','mean','p25','p75'}
    if not required.issubset(table.columns):
        raise ValueError(f'Annual summary requires columns {sorted(required)}.')
    if table.duplicated(['year','series']).any():
        raise ValueError('Duplicate annual summary rows.')
    years=sorted(table.year.unique())
    data={}
    for key in DISPLAY:
        rows=table[table.series==key].set_index('year').reindex(years)
        data[key]={field:rows[field].to_numpy(float) for field in ('mean','p25','p75')}
    return {'years':years,'series':data}

def plot_annual_summary(ctx):
    source=read_annual_summary(ctx.path('annual_summary','annual_summary.csv'))
    years=np.asarray(source['years'],dtype=float)
    if not np.array_equal(years,np.arange(1981,2020)):
        raise ValueError('Expected annual observations for 1981–2019.')
    trend={}
    for key,data in source['series'].items():
        y=np.asarray(data['mean'],dtype=float)
        lower,upper=np.asarray(data['p25']),np.asarray(data['p75'])
        if y.shape!=lower.shape or y.shape!=upper.shape or not np.isfinite(np.column_stack([y,lower,upper])).all():
            raise ValueError(f'Invalid annual summary for {key}.')
        if np.any(lower>upper):
            raise ValueError('25th percentile exceeds 75th percentile.')
        values=np.column_stack([years,y])
        result=ktaub(values)
        trend[key]={'sen_z_per_decade':float(result.sen*10),'mk_p':float(result.p),
                    'line':source_sen_line(values,result)[:,1]}
    if ctx.check_inputs:
        print('Fig4: valid annual means and quartiles for 1981–2019.')
        return
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.major.size': 3, 'ytick.major.size': 3, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 1, figsize=(180 / 25.4, 185 / 25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.135, right=0.976, top=0.9, bottom=0.153, hspace=0.3)
    fig.suptitle('Global basin-area-weighted mean (1981–2019)', fontsize=11, y=0.979)
    groups = [['SSI-12'], ['GHCD SPEI-12', 'ERA5 SPEI-12', 'JRA-3Q SPEI-12']]
    annotations = []
    for index, (ax, keys) in enumerate(zip(axes, groups)):
        ax.set_label(chr(ord('a') + index))
        ax.set_title('12-month SSI' if index == 0 else '12-month SPEI', pad=9)
        ax.annotate(chr(ord('a') + index), xy=(0, 1), xycoords='axes fraction', xytext=(-30, 9), textcoords='offset points', fontsize=11, weight='bold', va='bottom', ha='left', annotation_clip=False)
        ax.axhline(0, color='0.5', linestyle=(0, (3, 2)), linewidth=0.75, zorder=1)
        for key in keys:
            data = source['series'][key]
            display = DISPLAY[key]
            color = COLORS[display]
            ax.fill_between(years, data['p25'], data['p75'], color=color, alpha=0.14, linewidth=0, zorder=0)
            ax.plot(years, data['mean'], color=color, linewidth=1.65, zorder=3, label=display)
            ax.plot(years, trend[key]['line'], color=color, linewidth=1.3, linestyle=(0, (5, 3)), zorder=2)
        ax.set_xlim(1980, 2020)
        ax.set_ylim(-1.6, 2.15)
        ax.set_yticks(np.arange(-1.5, 2.01, 0.5))
        ax.set_xticks(np.arange(1980, 2021, 5))
        ax.set_ylabel('Area-weighted index (z units)', labelpad=6)
        ax.tick_params(axis='x', labelbottom=True)
        upper_envelope = max((max(source['series'][key]['p75']) for key in keys))
        text_ys = [1.98] if index == 0 else [1.98, 1.61, 1.24]
        assert min(text_ys) > upper_envelope + 0.14
        for key, text_y in zip(keys, text_ys):
            display = DISPLAY[key]
            value = trend[key]['sen_z_per_decade']
            signed = f'{value:+.4f}'.replace('-', '−')
            label = f"{display}: {signed} decade$^{{-1}}$, {p_text(trend[key]['mk_p'])}"
            ax.text(2019.7, text_y, label, ha='right', va='top', fontsize=8.7, color=COLORS[display], zorder=5)
            annotations.append(label)
    axes[1].set_xlabel('Year', labelpad=5)
    series_handles = [Line2D([], [], color=COLORS[k], linewidth=1.65, label=k) for k in ['12-month SSI', 'GHCD', 'ERA5', 'JRA-3Q']]
    fig.legend(handles=series_handles, loc='lower center', bbox_to_anchor=(0.54, 0.057), ncol=4, frameon=False, handlelength=2.8, columnspacing=1.7)
    style_handles = [Line2D([], [], color='0.4', linestyle=(0, (5, 3)), linewidth=1.3, label='Sen trend'), Patch(facecolor='0.75', alpha=0.4, edgecolor='none', label='Area-weighted 25th–75th percentile')]
    fig.legend(handles=style_handles, loc='lower center', bbox_to_anchor=(0.54, 0.017), ncol=2, frameon=False, fontsize=7.7, handlelength=2.6, columnspacing=1.8)
    save_figure(fig,ctx)
