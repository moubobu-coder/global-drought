"""Annual meteorological, SPEI and drought-area figures."""
from __future__ import annotations

import math
import string

import numpy as np
import pandas as pd

from .io import find_array, load_mat, read_series
from .style import save_figure

DATASETS = ("GHCD", "ERA5", "JRA-3Q")
FILE_DATASETS = ("HGD01", "ERA5", "JRA3Q")
COLORS = ((0.0, 0.0, 0.0), (0.0, 0.45, 0.74), (0.85, 0.33, 0.10))
CONTINENTS = ("Africa", "Asia", "Europe", "North America", "Oceania", "South America")
SOURCE_CONTINENTS = ("Asia", "Africa", "North America", "South America", "Antarctica", "Europe", "Oceania")
CLASSES = ("All", "Moderate", "Severe", "Extreme")
INDICES = ("SPEI", "SPEI-AEDclim")


def _years(ctx):
    start = int(ctx.config.get("start_year", 1973))
    end = int(ctx.config.get("end_year", 2024))
    if end < start:
        raise ValueError("end_year must be at least start_year")
    return np.arange(start, end + 1)


def _source_names(ctx):
    """Use metadata when supplied, otherwise the seven-column source order."""
    supplied = ctx.config.get("continent_names")
    if supplied is not None:
        names = tuple(str(v).strip() for v in supplied)
    else:
        mask = ctx.path("continent_mask", "mask/continent_0-359.75_90--90_0.25deg.mat")
        if mask.is_file():
            m = load_mat(mask)
            value = m.get("continent_name")
            names = _matlab_names(value) if value is not None else SOURCE_CONTINENTS
        else:
            names = SOURCE_CONTINENTS
    if len(set(n.casefold() for n in names)) != len(names):
        raise ValueError("continent_names must be unique")
    missing = set(CONTINENTS) - set(names)
    if missing:
        raise ValueError(f"Missing continent names: {sorted(missing)}")
    return names


def _matlab_names(value):
    arr = np.asarray(value)
    if arr.dtype.kind in "US" and arr.ndim == 2 and all(len(str(x)) == 1 for x in arr.ravel()):
        return tuple("".join(map(str, row)).strip() for row in arr)
    names = []
    for item in arr.ravel():
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.item()
        if isinstance(item, np.ndarray):
            item = "".join(str(x) for x in item.ravel())
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        names.append(str(item).strip())
    return tuple(names)


def _shape(values, expected, path):
    values = np.asarray(values, dtype=float)
    if values.shape != tuple(expected):
        raise ValueError(f"{path.name}: expected shape {tuple(expected)}, found {values.shape}")
    return values


def _forcing_path(ctx, dataset, variable, continental, years):
    directory = ctx.path("forcing_series", "F_series")
    domain = "GLcont" if continental else "GL"
    suffix = "_ERA5" if dataset == "HGD01" and variable == "PET" else ""
    return directory / f"{dataset}_{variable}_{domain}_series_{years[0]}-{years[-1]}_Mo2Yr{suffix}.mat"


def _forcing(ctx, variable, dataset_index, years, names=None):
    dataset = FILE_DATASETS[dataset_index]
    path = _forcing_path(ctx, dataset, variable, names is not None, years)
    shape = (len(years), len(names)) if names is not None else (len(years),)
    values = _shape(read_series(path, dataset + "_series_yr"), shape, path)
    # The annual VPD files are in kPa; the figure reports hPa.
    if variable == "VPD":
        values = values * 10
    return values


def _trend(years, values, alpha=.05):
    from .stats import ktaub, source_sen_line

    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    fit = np.full(values.shape, np.nan)
    if np.count_nonzero(valid) < 3:
        return np.nan, False, fit
    data = np.column_stack((np.asarray(years)[valid], values[valid]))
    result = ktaub(data, alpha)
    fit[valid] = source_sen_line(data, result)[:, 1]
    return result.sen, bool(result.h), fit


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.linewidth": .75, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "axes.grid": False})
    return plt


def _axis(ax, years, ylabel, xlabel=False):
    ax.set_xlim(years[0], years[-1])
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="in", labelsize=8.5)
    if xlabel:
        ax.set_xlabel("Year")
    for spine in ax.spines.values():
        spine.set_visible(True)


def _panel(ax, label, height=1.03):
    ax.text(.012, height, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom")


def _annotation(ax, numbers, significant, ids=(0, 1, 2), unit="", xy=(.015, .06), size=8, trim=False):
    """Keep dataset colours and the source h flags in trend annotations."""
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

    pieces = []
    for position, dataset_id in enumerate(ids):
        value = numbers[dataset_id]
        text = f"{value:.2f}"
        if trim:
            text = text.rstrip("0").rstrip(".")
        if significant[dataset_id]:
            text += r"$^{*}$"
        if position < len(ids) - 1:
            text += ", "
        pieces.append(TextArea(text, textprops={"color": COLORS[dataset_id], "fontsize": size,
                                               "fontweight": "bold"}))
    if unit:
        pieces.append(TextArea(" " + unit, textprops={"color": "black", "fontsize": size,
                                                   "fontweight": "bold"}))
    box = HPacker(children=pieces, align="baseline", pad=0, sep=0)
    artist = AnnotationBbox(box, xy, xycoords=ax.transAxes, box_alignment=(0, 0),
                           frameon=False, pad=0, annotation_clip=False)
    ax.add_artist(artist)


def _checked(ctx, figure_id):
    if ctx.check_inputs:
        print(f"{figure_id} inputs checked.")
        return True
    return False


def plot_fig1(ctx):
    """Six annual driver series; panel e is precipitation and panel f is AED."""
    years = _years(ctx)
    variables = ("t2m", "ssrd", "wind", "VPD", "tp", "PET")
    labels = ("Ta (°C)", r"Rs (W m$^{-2}$)", r"WS (m s$^{-1}$)", "VPD (hPa)",
              r"P (mm yr$^{-1}$)", r"AED (mm yr$^{-1}$)")
    units = (r"(°C dec$^{-1}$)", r"(W m$^{-2}$ dec$^{-1}$)", r"(m s$^{-1}$ dec$^{-1}$)",
             r"(hPa dec$^{-1}$)", r"(mm yr$^{-1}$ dec$^{-1}$)", r"(mm yr$^{-1}$ dec$^{-1}$)")
    series = {(iv, dataset_id): _forcing(ctx, var, dataset_id, years)
              for iv, var in enumerate(variables) for dataset_id in range(3)
              if not (iv == 0 and dataset_id == 0)}
    if _checked(ctx, "Fig1"):
        return
    plt = _mpl()
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 8.2))
    fig.subplots_adjust(left=.09, right=.985, bottom=.08, top=.96, wspace=.22, hspace=.22)
    for iv, ax in enumerate(axes.flat):
        ids = (1, 2) if iv == 0 else (0, 1, 2)
        slopes, sig = np.full(3, np.nan), np.zeros(3, dtype=bool)
        for dataset_id in ids:
            values = series[(iv, dataset_id)]
            ax.plot(years, values, color=COLORS[dataset_id], lw=1.2, label=DATASETS[dataset_id])
            # The global GHCD source omits its trend if any year is missing.
            if dataset_id == 0 and np.isnan(values).any():
                fit = np.full(values.shape, np.nan)
            else:
                slopes[dataset_id], sig[dataset_id], fit = _trend(years, values)
            ax.plot(years, fit, "--", color=COLORS[dataset_id], lw=.5)
        _axis(ax, years, labels[iv], iv >= 4)
        ax.set_xlim(years[0] - 1, years[-1] + 1)
        _panel(ax, string.ascii_lowercase[iv], 1.015)
        if iv == 1:
            ax.legend(frameon=False, fontsize=9, loc="best")
        xy = (.01, .93) if iv == 4 else (.01, .05)
        if iv in (1, 3):
            xy = (.40, .05)
        _annotation(ax, slopes * 10, sig, ids, units[iv], xy, size=9, trim=True)
    save_figure(fig, ctx)
    plt.close(fig)


def plot_continental_forcing(ctx, figure_id):
    """Shared six-continent driver-series layout for S2, S3 and S4."""
    definitions = {
        "S2": (("t2m", "ssrd", "wind"), ("Ta", "Rs", "WS"),
               ("Ta (°C)", r"Rs (W m$^{-2}$)", r"WS (m s$^{-1}$)"),
               (r"°C decade$^{-1}$", r"W m$^{-2}$ decade$^{-1}$", r"m s$^{-1}$ decade$^{-1}$")),
        "S3": (("RH", "VPD"), ("RH", "VPD"), ("RH (%)", "VPD (hPa)"),
               (r"% decade$^{-1}$", r"hPa decade$^{-1}$")),
        "S4": (("tp", "PET"), ("P", "AED"), (r"P (mm yr$^{-1}$)", r"AED (mm yr$^{-1}$)"),
               (r"mm yr$^{-1}$ decade$^{-1}$", r"mm yr$^{-1}$ decade$^{-1}$")),
    }
    variables, titles, labels, units = definitions[figure_id]
    years, names = _years(ctx), _source_names(ctx)
    series = {(iv, dataset_id): _forcing(ctx, var, dataset_id, years, names)
              for iv, var in enumerate(variables) for dataset_id in range(3)
              if not (figure_id == "S2" and iv == 0 and dataset_id == 0)}
    if _checked(ctx, figure_id):
        return
    plt = _mpl()
    ncols = len(variables)
    fig, axes = plt.subplots(6, ncols, figsize=(12.6 if ncols == 3 else 9.45, 11.42))
    fig.subplots_adjust(left=.075, right=.985, bottom=.055, top=.955,
                        wspace=.24 if ncols == 3 else .21, hspace=.52)
    for ic, continent in enumerate(CONTINENTS):
        source_index = names.index(continent)
        for iv in range(ncols):
            ax = axes[ic, iv]
            ids = (1, 2) if figure_id == "S2" and iv == 0 else (0, 1, 2)
            slopes, sig = np.full(3, np.nan), np.zeros(3, dtype=bool)
            for dataset_id in ids:
                values = series[(iv, dataset_id)][:, source_index]
                ax.plot(years, values, color=COLORS[dataset_id], lw=1.05, label=DATASETS[dataset_id])
                slopes[dataset_id], sig[dataset_id], fit = _trend(years, values)
                ax.plot(years, fit, "--", color=COLORS[dataset_id], lw=.6)
            _axis(ax, years, labels[iv], ic == 5)
            ax.set_title(f"{continent} - {titles[iv]}", fontsize=9.5, fontweight="normal", pad=5)
            _panel(ax, string.ascii_lowercase[ic * ncols + iv])
            if ic == 0:
                ax.text(1, 1.16, "Trend unit: " + units[iv], transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7.1, fontweight="bold")
            if figure_id == "S2":
                xy = (.63, .05) if iv < 2 else (.015, .05)
            elif figure_id == "S3":
                xy = (.64, .075) if iv == 1 else (.015, .075)
                if iv == 0 and ic == 1:
                    xy = (.015, .85)
            else:
                xy = (.63, .075) if iv == 1 else (.015, .075)
            _annotation(ax, slopes * 10, sig, ids, xy=xy, size=7.5)
            if ic == 0 and iv == 0:
                from matplotlib.lines import Line2D
                handles = [Line2D([], [], color=c, lw=1.05, label=n) for c, n in zip(COLORS, DATASETS)]
                ax.legend(handles=handles, frameon=False, fontsize=7,
                          ncol=1 if figure_id == "S2" else 3, loc="upper left")
    save_figure(fig, ctx)
    plt.close(fig)


def _read_table(path, columns, keys):
    table = pd.read_csv(path)
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if table.duplicated(keys).any():
        raise ValueError(f"{path.name}: duplicate records for {keys}")
    for col in set(columns) - {"region", "dataset", "index", "drought_class"}:
        table[col] = pd.to_numeric(table[col], errors="raise")
    return table


def _area_tables(ctx, annual=True):
    source = ctx.path("source_data", "source_data")
    trends = _read_table(source / "trend_statistics_long.csv",
                         ("region", "dataset", "index", "drought_class", "trend_pp_decade", "significant"),
                         ["region", "dataset", "index", "drought_class"])
    if not trends.significant.isin([0, 1]).all():
        raise ValueError("significant must contain the source h flag, 0 or 1")
    areas = None
    if annual:
        areas = _read_table(source / "annual_area_long.csv",
                           ("region", "dataset", "index", "drought_class", "year", "area_pct", "fit_value_pct"),
                           ["region", "dataset", "index", "drought_class", "year"])
    return areas, trends


def _selection(table, region, dataset, index, drought_class):
    take = ((table.region == region) & (table.dataset == dataset) & (table["index"] == index)
            & (table.drought_class == drought_class))
    return table.loc[take]


def _one_trend(trends, region, dataset, index, drought_class):
    selected = _selection(trends, region, dataset, index, drought_class)
    if len(selected) != 1:
        raise ValueError(f"Expected one trend for {region}, {dataset}, {index}, {drought_class}; found {len(selected)}")
    return selected.iloc[0]


def plot_fig3(ctx):
    """Global annual drought-area series from the paired source-data tables."""
    years = _years(ctx)
    areas, trends = _area_tables(ctx)
    selected = {}
    for drought_class in CLASSES:
        for index in INDICES:
            for dataset in DATASETS:
                annual = _selection(areas, "Global", dataset, index, drought_class).sort_values("year")
                if not np.array_equal(annual.year.to_numpy(), years):
                    raise ValueError(f"Incomplete annual record for Global, {dataset}, {index}, {drought_class}")
                selected[(drought_class, index, dataset)] = (annual, _one_trend(trends, "Global", dataset, index, drought_class))
    if _checked(ctx, "Fig3"):
        return
    plt = _mpl()
    fig, axes = plt.subplots(4, 2, figsize=(9.84, 8.66))
    fig.subplots_adjust(left=.085, right=.985, bottom=.07, top=.955, wspace=.20, hspace=.40)
    bounds = ("< −1.0", "−1.5 ≤ {index} < −1.0", "−2.0 ≤ {index} < −1.5", "< −2.0")
    for ic, drought_class in enumerate(CLASSES):
        row = areas.loc[(areas.region == "Global") & (areas.drought_class == drought_class), "area_pct"]
        y_max = max(5, math.ceil(1.08 * row.max() / 5) * 5)
        for ii, index in enumerate(INDICES):
            ax = axes[ic, ii]
            numbers, sig = np.zeros(3), np.zeros(3, dtype=bool)
            for dataset_id, dataset in enumerate(DATASETS):
                annual, trend = selected[(drought_class, index, dataset)]
                ax.plot(annual.year, annual.area_pct, color=COLORS[dataset_id], lw=1.2, label=dataset)
                ax.plot(annual.year, annual.fit_value_pct, "--", color=COLORS[dataset_id], lw=.6)
                numbers[dataset_id] = trend.trend_pp_decade
                sig[dataset_id] = bool(trend.significant)
            condition = bounds[ic].format(index=index) if "{index}" in bounds[ic] else f"{index} {bounds[ic]}"
            ax.set_title(f"{drought_class} drought ({condition})", fontsize=10.0, fontweight="bold", pad=5)
            _axis(ax, years, "Area in drought (%)", ic == 3)
            ax.set_ylim(0, y_max)
            _panel(ax, string.ascii_lowercase[2 * ic + ii], 1.12)
            _annotation(ax, numbers, sig, xy=(.015, .88), size=9)
            if ic == 0 and ii == 0:
                ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3,
                          handlelength=1.6, columnspacing=.7)
            if ic == 0 and ii == 1:
                ax.text(.985, .985, r"Trend unit: % decade$^{-1}$", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, fontweight="bold")
    save_figure(fig, ctx)
    plt.close(fig)


def plot_s6(ctx):
    """December SPEI-12 continent series and the stored source trend flags."""
    years, names = _years(ctx), _source_names(ctx)
    directory = ctx.path("spei_series", "series/GPCC")
    products = {}
    for product in ("HGD01-ERA5", "HGD01-JRA3Q"):
        base = f"{product}_SPEI12_GLcont_series_{years[0]}-{years[-1]}"
        series_path, trend_path = directory / (base + "_12thYr.mat"), directory / (base + "_trend_12thYr.mat")
        series = _shape(find_array(load_mat(series_path), ["SPEI_series_Yr"]), (len(years), 6, len(names)), series_path)
        trend = load_mat(trend_path)
        sen = _shape(find_array(trend, ["SPEI_sen_series_Yr"]), (6, len(names)), trend_path)
        h = _shape(find_array(trend, ["SPEI_p_series_Yr"]), (6, len(names)), trend_path)
        products[product] = (series, sen, h)
    if _checked(ctx, "S6"):
        return
    plt = _mpl()
    fig, axes = plt.subplots(3, 2, figsize=(9.45, 7.09))
    fig.subplots_adjust(left=.08, right=.985, bottom=.07, top=.96, wspace=.20, hspace=.28)
    plot_names = tuple(n for n in names if "antarctica" not in n.casefold() and "南极" not in n)
    if len(plot_names) != 6:
        raise ValueError("S6 needs six continents after excluding Antarctica")
    era5, jra3q = products["HGD01-ERA5"], products["HGD01-JRA3Q"]
    for ic, (ax, continent) in enumerate(zip(axes.flat, plot_names)):
        k = names.index(continent)
        values = (era5[0][:, 0, k], era5[0][:, 5, k], jra3q[0][:, 5, k])
        numbers = np.array((era5[1][0, k], era5[1][5, k], jra3q[1][5, k])) * 10
        sig = np.array((era5[2][0, k], era5[2][5, k], jra3q[2][5, k])) == 1
        for dataset_id, y in enumerate(values):
            ax.plot(years, y, color=COLORS[dataset_id], lw=1.2, label=DATASETS[dataset_id])
            _, _, fit = _trend(np.arange(1, len(years) + 1), y)
            ax.plot(years, fit, "--", color=COLORS[dataset_id], lw=.5)
        ax.axhline(0, ls=":", color=".55", lw=.65)
        _axis(ax, years, "SPEI (z-unit)", ic >= 4)
        ax.set_title(continent, fontsize=11, fontweight="bold")
        _annotation(ax, numbers, sig, unit=r"(z-unit decade$^{-1}$)", xy=(.02, .04), size=8.4)
        if ic == 0:
            ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    save_figure(fig, ctx)
    plt.close(fig)


def plot_s8(ctx):
    """Continental drought-area trends; colour is dataset and shape is index."""
    _, trends = _area_tables(ctx, annual=False)
    trends = trends.loc[trends.region != "Global"]
    selected = {(region, drought_class, dataset, index): _one_trend(trends, region, dataset, index, drought_class)
                for region in CONTINENTS for drought_class in CLASSES for dataset in DATASETS for index in INDICES}
    if _checked(ctx, "S8"):
        return
    plt = _mpl()
    from matplotlib.lines import Line2D
    values = trends.trend_pp_decade.to_numpy()
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite continental drought-area trends")
    xpad = max(.25, .06 * (values.max() - values.min()))
    xlims = (min(values.min(), 0) - xpad, max(values.max(), 0) + xpad)
    fig, axes = plt.subplots(3, 2, figsize=(9.06, 7.09))
    fig.subplots_adjust(left=.10, right=.98, bottom=.09, top=.95, wspace=.22, hspace=.25)
    for ir, (ax, region) in enumerate(zip(axes.flat, CONTINENTS)):
        ax.axvline(0, ls=":", color=".45", lw=.7)
        for ic, drought_class in enumerate(CLASSES):
            for dataset_id, dataset in enumerate(DATASETS):
                for ii, index in enumerate(INDICES):
                    row = selected[(region, drought_class, dataset, index)]
                    y = 4 * ic + dataset_id + (-.12, .12)[ii]
                    x = row.trend_pp_decade
                    ax.scatter(x, y, s=42, marker=("o", "^")[ii], color=COLORS[dataset_id], alpha=.4)
                    if row.significant:
                        ax.text(x + .012 * (xlims[1] - xlims[0]), y, "*", color=COLORS[dataset_id],
                                fontsize=9, fontweight="bold", va="center")
        ax.set_xlim(xlims)
        ax.set_ylim(14.7, -.7)
        ax.set_yticks((1, 5, 9, 13), CLASSES)
        ax.set_title(region, fontweight="bold")
        _panel(ax, string.ascii_lowercase[ir])
        ax.tick_params(direction="in", labelsize=9)
        if ir >= 4:
            ax.set_xlabel(r"Drought-area trend (percentage points decade$^{-1}$)", fontsize=8.7)
    handles = [Line2D([], [], color=c, lw=2.2, label=n) for c, n in zip(COLORS, DATASETS)]
    handles += [Line2D([], [], color="black", marker=m, ls="none", label=n, markersize=5.5)
                for m, n in zip(("o", "^"), INDICES)]
    axes.flat[0].legend(handles=handles, frameon=False, ncol=2, fontsize=8, loc="best")
    save_figure(fig, ctx)
    plt.close(fig)
