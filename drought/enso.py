"""DJF ENSO correlations and area-weighted agreement statistics."""
from __future__ import annotations
import csv
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
from .style import save_figure

THRESHOLDS = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
BOUNDS = np.array([-0.8, -0.4, -0.3, -0.1, 0.1, 0.3, 0.4, 0.8])
MAP_COLORS = ['#543005', '#8c510a', '#dfc27d', '#f5f5f5', '#80cdc1', '#01665e', '#003c30']
PAIR_DEFINITIONS = [('GHCD-ERA5',0,1,'#2474A6','o'), ('GHCD-JRA-3Q',0,2,'#D67A30','s'), ('ERA5-JRA-3Q',1,2,'#696969','^')]
NAMES = ['GHCD', 'ERA5', 'JRA-3Q']

def djf_correlations(spei, time, nino34, winter_end_years=None):
    """Pearson r of same-season Dec/Jan/Feb means on a complete common domain.

    spei is dataset x month x cell; nino34 has one value for each time entry.
    """
    import pandas as pd
    dates = pd.DatetimeIndex(time)
    values = np.asarray(spei, dtype=float)
    nino34 = np.asarray(nino34, dtype=float)
    if values.ndim != 3 or values.shape[1] != len(dates) or nino34.shape != (len(dates),):
        raise ValueError('Expected SPEI dataset x month x cell and matching monthly dates/ENSO index.')
    keys = {(d.year, d.month): i for i, d in enumerate(dates)}
    if len(keys) != len(dates):
        raise ValueError('Duplicated monthly dates.')
    years = np.arange(1974, 2025) if winter_end_years is None else np.asarray(winter_end_years)
    selected = np.array([[keys[(int(y)-1,12)],keys[(int(y),1)],keys[(int(y),2)]] for y in years])
    winter = values[:, selected, :].mean(axis=2)
    x = nino34[selected].mean(axis=1)
    common = np.isfinite(winter).all(axis=(0,1))
    if not np.isfinite(x).all() or len(x)<3 or np.var(x)==0:
        raise ValueError('Invalid winter ENSO series.')
    r = np.full((values.shape[0],values.shape[2]),np.nan)
    xc=x-x.mean()
    for i in range(values.shape[0]):
        yc=winter[i,:,common].T
        yc=yc-yc.mean(axis=0)
        denom=np.sqrt(np.sum(xc*xc)*np.sum(yc*yc,axis=0))
        r[i,common]=np.divide(xc@yc,denom,out=np.full(denom.shape,np.nan),where=denom>0)
    return np.clip(r,-1,1),common,years

def compute_statistics(r: np.ndarray, w: np.ndarray) -> dict:
    threshold_rows = []
    for t in THRESHOLDS:
        common = np.max(np.abs(r), axis=0) >= t
        common_weight = float(w[common].sum())
        for pair, i, j, _, _ in PAIR_DEFINITIONS:
            a, b = (r[i], r[j])
            union = (np.abs(a) >= t) | (np.abs(b) >= t)
            shared = (np.abs(a) >= t) & (np.abs(b) >= t) & (a * b > 0)
            union_weight = float(w[union].sum())
            shared_weight = float(w[shared].sum())
            if union_weight <= 0 or common_weight <= 0:
                raise ValueError('Empty threshold denominator')
            threshold_rows.append({'threshold': float(t), 'pair': pair, 'union_cells': int(union.sum()), 'union_area_weight': union_weight, 'same_sign_strong_cells': int(shared.sum()), 'same_sign_strong_area_weight': shared_weight, 'overlap_percent': 100 * shared_weight / union_weight, 'common_strong_cells': int(common.sum()), 'common_strong_area_weight': common_weight, 'mae': float(np.sum(w[common] * np.abs(a[common] - b[common])) / common_weight), 'sign_agreement_percent': float(100 * w[common & (a * b > 0)].sum() / common_weight)})
    reproduction_rows = []
    for label, direction in [('negative', -1), ('positive', 1)]:
        reference = direction * r[0] >= 0.3
        reference_weight = float(w[reference].sum())
        for j in (1, 2):
            reproduced = reference & (direction * r[j] >= 0.3)
            reproduced_weight = float(w[reproduced].sum())
            reproduction_rows.append({'threshold': 0.3, 'ghcd_sign': label, 'comparison': NAMES[j], 'reference_cells': int(reference.sum()), 'reference_area_weight': reference_weight, 'reproduced_cells': int(reproduced.sum()), 'reproduced_area_weight': reproduced_weight, 'reproduction_percent': 100 * reproduced_weight / reference_weight})
    area = {name: {'negative_r_le_minus_0_3': float(100 * w[x <= -0.3].sum()), 'positive_r_ge_plus_0_3': float(100 * w[x >= 0.3].sum())} for name, x in zip(NAMES, r, strict=True)}
    return {'metadata': {'dataset_order': NAMES, 'thresholds': THRESHOLDS.tolist(), 'common_land_cells': r.shape[1], 'n_winters': 51, 'winter_period': '1973/74-2023/24', 'weighting': 'cos(latitude), normalized to sum to one on common valid land', 'b_denominator': 'Pair-specific union: either member has abs(r) >= threshold', 'd_denominator': 'Common domain: at least one of the three has abs(r) >= threshold', 'f_denominator': 'GHCD negative and positive domains separately at abs(r) >= 0.3', 'domain_rule': 'Forcing mask2 == 0 intersected with complete common valid input data'}, 'threshold_metrics': threshold_rows, 'ghcd_reproduction': reproduction_rows, 'map_area_fractions': area}

def display_slab(values: np.ndarray, lat_i: np.ndarray, lon_i: np.ndarray):
    slab = np.full((721, 1440), np.nan, dtype=np.float64)
    slab[lat_i, lon_i] = values
    return np.ma.masked_invalid(np.roll(slab[120:601, :], 720, axis=1))

def panel_title(ax, letter: str, title: str) -> None:
    ax.text(0.01, 1.055, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    ax.text(0.5, 1.055, title, transform=ax.transAxes, fontsize=7.2, ha='center', va='bottom')

def style_quantitative(ax) -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.6)
    ax.tick_params(axis='both', labelsize=6, length=2.5, width=0.6, pad=2)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linewidth=0.35, color='#dedede')

def plot_enso(ctx):
    r = np.load(ctx.path('enso_correlations','ENSO/r_raw.npy'),allow_pickle=False).astype(float)
    with np.load(ctx.path('enso_grid','ENSO/grid_index.npz'),allow_pickle=False) as grid:
        lat_i=grid['lat_index'].astype(int)
        lon_i=grid['lon_index'].astype(int)
    if r.shape != (3,len(lat_i)) or len(lon_i)!=len(lat_i) or not np.isfinite(r).all():
        raise ValueError('Expected GHCD, ERA5, JRA-3Q correlation rows and matching grid indices.')
    if np.any(np.abs(r)>1) or np.any((lat_i<120)|(lat_i>600)) or np.any((lon_i<0)|(lon_i>=1440)):
        raise ValueError('Invalid correlation values or 0.25-degree grid coordinates.')
    if len(np.unique(lat_i*1440+lon_i))!=len(lat_i):
        raise ValueError('Duplicated grid cells.')
    if 'forcing_mask' in ctx.config.get('paths',{}):
        from .io import load_mat,find_array
        mask=find_array(load_mat(ctx.path('forcing_mask','mask/forcing_mask.mat')),'mask2')
        if mask.shape!=(1440,721) or not np.all(mask[lon_i,lat_i]==0):
            raise ValueError('Correlation grid is outside Forcing mask2==0.')
    w=np.cos(np.deg2rad(90-0.25*lat_i))
    w/=w.sum()
    statistics=compute_statistics(r,w)
    if ctx.check_inputs:
        print(f'S9: valid correlation input for {r.shape[1]} cells.')
        return
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7, 'axes.linewidth': 0.6, 'legend.frameon': False})
    fig = plt.figure(figsize=(7.2, 6.4), facecolor='white')
    fig_width, fig_height = fig.get_size_inches()
    projection = ccrs.Robinson(central_longitude=0)
    plate = ccrs.PlateCarree()
    y_limit = abs(projection.transform_point(0, 60, plate)[1])
    map_aspect = (projection.x_limits[1] - projection.x_limits[0]) / (2 * y_limit)
    left_x, left_width, right_x, right_width = (0.055, 0.525, 0.69, 0.285)
    panel_height = left_width * fig_width / map_aspect / fig_height
    row_bottoms = [0.745, 0.437, 0.129]
    map_axes, stat_axes = ([], [])
    cmap = ListedColormap(MAP_COLORS)
    cmap.set_bad((1, 1, 1, 0))
    cmap.set_under(MAP_COLORS[0])
    cmap.set_over(MAP_COLORS[-1])
    norm = BoundaryNorm(BOUNDS, cmap.N, clip=False)
    for row, (name, i, letter) in enumerate([('ERA5', 1, 'a'), ('JRA-3Q', 2, 'c'), ('GHCD', 0, 'e')]):
        ax = fig.add_axes((left_x, row_bottoms[row], left_width, panel_height), projection=projection)
        ax.set_facecolor('white')
        ax.set_xlim(*projection.x_limits)
        ax.set_ylim(-y_limit, y_limit)
        ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor='#e8e8e8', edgecolor='none', zorder=0)
        image = ax.imshow(display_slab(r[i], lat_i, lon_i), extent=(-180.125, 179.875, -60.125, 60.125), transform=plate, origin='upper', interpolation='nearest', regrid_shape=1800, cmap=cmap, norm=norm, zorder=1)
        coast = ax.coastlines(resolution='110m', color='#303030', linewidth=0.35, zorder=2)
        coast.set_rasterized(True)
        ax.spines['geo'].set_linewidth(0.5)
        ax.spines['geo'].set_color('#303030')
        ax.spines['geo'].set_rasterized(True)
        panel_title(ax, letter, name)
        fractions = statistics['map_area_fractions'][name]
        ax.text(0.075, 0.27, 'Area fraction (%)', transform=ax.transAxes, fontsize=6, color='#303030', ha='left', va='center', zorder=5)
        for y, label, value, color in [(0.19, 'r ≤ −0.3', fractions['negative_r_le_minus_0_3'], MAP_COLORS[1]), (0.11, 'r ≥ 0.3', fractions['positive_r_ge_plus_0_3'], MAP_COLORS[5])]:
            ax.text(0.075, y, f'{label}   {value:.1f}', transform=ax.transAxes, fontsize=6, color=color, ha='left', va='center', zorder=5)
        for y, label in [(0.82, '50°N'), (0.5, '0°'), (0.18, '50°S')]:
            ax.text(-0.018, y, label, transform=ax.transAxes, fontsize=6, ha='right', va='center', color='#303030')
        if row == 2:
            for x, label in [(0.25, '120°W'), (0.5, '0°'), (0.75, '120°E')]:
                ax.text(x, -0.055, label, transform=ax.transAxes, fontsize=6, ha='center', va='top', color='#303030')
        map_axes.append(ax)
        stat = fig.add_axes((right_x, row_bottoms[row], right_width, panel_height))
        style_quantitative(stat)
        stat_axes.append(stat)
    plotted = {}
    for ax, metric, letter, title in [(stat_axes[0], 'overlap_percent', 'b', 'Same-sign strong-correlation overlap'), (stat_axes[1], 'mae', 'd', 'Agreement in correlation magnitude')]:
        for pair, _, _, color, marker in PAIR_DEFINITIONS:
            rows = [x for x in statistics['threshold_metrics'] if x['pair'] == pair]
            values = [x[metric] for x in rows]
            ax.plot(THRESHOLDS, values, marker=marker, markersize=3.2, linewidth=1.2, color=color, label=pair.replace('-ERA5', '–ERA5').replace('-JRA', '–JRA'), zorder=3)
            plotted.setdefault(letter, {})[pair] = values
        ax.set_xlim(0.075, 0.525)
        ax.set_xticks(THRESHOLDS)
        ax.set_xlabel('Correlation threshold (|r|)', fontsize=6.5, labelpad=3)
        for tick in ax.get_xticklabels():
            if tick.get_text() == '0.3':
                tick.set_fontweight('bold')
        panel_title(ax, letter, title)
    stat_axes[0].set_ylim(0, 100)
    stat_axes[0].set_yticks([0, 25, 50, 75, 100])
    stat_axes[0].set_ylabel('Overlap (%)', fontsize=6.5, labelpad=3)
    stat_axes[0].axvline(0.3, ymin=0, ymax=0.73, color='#999999', linewidth=0.65, linestyle=(0, (3, 3)), zorder=1)
    stat_axes[0].legend(loc='upper right', fontsize=6, handlelength=1.8, labelspacing=0.3, borderaxespad=0.3)
    stat_axes[1].set_ylim(0, 0.18)
    stat_axes[1].set_yticks([0, 0.05, 0.1, 0.15])
    stat_axes[1].set_ylabel('Mean absolute difference in r', fontsize=6.5, labelpad=3)
    stat_axes[1].axvline(0.3, color='#999999', linewidth=0.65, linestyle=(0, (3, 3)), zorder=1)
    ax = stat_axes[2]
    categories = ['negative', 'positive']
    for name, color, offset in [('ERA5', '#2474A6', -0.16), ('JRA-3Q', '#D67A30', 0.16)]:
        values = [next((x['reproduction_percent'] for x in statistics['ghcd_reproduction'] if x['ghcd_sign'] == sign and x['comparison'] == name)) for sign in categories]
        bars = ax.bar(np.arange(2) + offset, values, width=0.28, color=color, label=name, zorder=3)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2.2, f'{value:.1f}', ha='center', va='bottom', fontsize=6, color='#303030')
        plotted.setdefault('f', {})[name] = values
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([0, 1], ['Negative\nr ≤ −0.3', 'Positive\nr ≥ 0.3'])
    ax.set_ylabel('GHCD area reproduced (%)', fontsize=6.5, labelpad=3)
    ax.legend(loc='upper center', ncol=2, fontsize=6, columnspacing=1, handlelength=1.3, borderaxespad=0.1)
    panel_title(ax, 'f', 'Reproduction of GHCD correlations')
    cax = fig.add_axes((0.105, 0.054, 0.425, 0.018))
    cb = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=BOUNDS, spacing='uniform', drawedges=True, extend='both', extendfrac=0.055)
    cb.set_label('Pearson correlation coefficient (r)', fontsize=7, labelpad=3)
    cb.ax.tick_params(labelsize=6, length=2, pad=2)
    save_figure(fig,ctx)
    (ctx.output/f'{ctx.figure_id}_statistics.json').write_text(json.dumps(statistics,indent=2),encoding='utf-8')
