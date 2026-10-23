"""Global trend maps, pressure-level RH series and station maps."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import string

import numpy as np

from .io import load_mat, find_array
from .maps import map_panel, projection
from .stats import ktaub, source_sen_line
from .style import save_figure


COLORS = ("black", (0.00, 0.45, 0.74), (0.85, 0.33, 0.10))
MASK_NAME = "Forcing_GL_0-359.75_90--90_0.25deg.mat"
YEAR_TAG = "1973-2024"


def _require(paths):
    paths = list(dict.fromkeys(Path(p) for p in paths))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
    return paths


def _checked(ctx, paths, detail=""):
    if not ctx.check_inputs:
        return False
    print(f"Validated {len(paths)} input files" + (f"; {detail}" if detail else ""))
    return True


def _vector(mat, names):
    value = np.asarray(find_array(mat, names), dtype=float).squeeze()
    if value.ndim != 1:
        raise ValueError(f"Expected vector {names}; found shape {value.shape}")
    return value


def _mat_grid(mat):
    lon = None
    lat = None
    for key in ("lon", "longitude", "lon_raw"):
        if key in mat:
            lon = np.asarray(mat[key], dtype=float).squeeze()
            break
    for key in ("lat", "latitude"):
        if key in mat:
            lat = np.asarray(mat[key], dtype=float).squeeze()
            break
    if lon is None:
        lon = np.arange(0, 360, 0.25)
    if lat is None:
        lat = np.arange(90, -90.01, -0.25)
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("MAT longitude and latitude coordinates must be one-dimensional")
    return lon, lat


def _lat_lon(array, lon, lat, label, orientation="lon_lat"):
    data = np.asarray(array, dtype=float).squeeze()
    if data.ndim != 2:
        raise ValueError(f"{label}: expected two-dimensional grid; found {data.shape}")
    if data.shape == (len(lon), len(lat)) and orientation == "lon_lat":
        return data.T
    if data.shape == (len(lat), len(lon)):
        return data
    if data.shape == (len(lon), len(lat)):
        return data.T
    raise ValueError(f"{label}: grid {data.shape} does not match {len(lat)} latitudes and {len(lon)} longitudes")


def _mat_trend(path):
    mat = load_mat(path)
    lon, lat = _mat_grid(mat)
    slope = _lat_lon(find_array(mat, ["senmap"]), lon, lat, f"{path.name}:senmap")
    flags = _lat_lon(find_array(mat, ["pmap"]), lon, lat, f"{path.name}:pmap")
    sig = np.isfinite(flags) & (flags == 1) & np.isfinite(slope)
    return lon, lat, slope, sig


def _mask(ctx):
    path = ctx.path("forcing_mask", f"mask/{MASK_NAME}")
    _require([path])
    mat = load_mat(path)
    lon, lat = _mat_grid(mat)
    first = _lat_lon(find_array(mat, ["mask1"]), lon, lat, "mask1")
    second = None
    if "mask2" in mat:
        second = _lat_lon(find_array(mat, ["mask2"]), lon, lat, "mask2")
    return path, lon, lat, first, second


def _on_grid(mask, mask_lon, mask_lat, lon, lat, outside=1):
    from scipy.interpolate import RegularGridInterpolator

    source_lon = (np.asarray(mask_lon) + 180) % 360 - 180
    ix = np.argsort(source_lon)
    iy = np.argsort(mask_lat)
    field = np.asarray(mask)[np.ix_(iy, ix)]
    interpolation = RegularGridInterpolator(
        (np.asarray(mask_lat)[iy], source_lon[ix]), field,
        method="nearest", bounds_error=False, fill_value=outside,
    )
    x, y = np.meshgrid((np.asarray(lon) + 180) % 360 - 180, lat)
    return interpolation(np.column_stack((y.ravel(), x.ravel()))).reshape(y.shape)


def _label(ax, letter, fontsize=10):
    ax.text(-0.025, 1.025, letter, transform=ax.transAxes, ha="right",
            va="bottom", weight="bold", fontsize=fontsize)


def _area_text(ax, drying, wetting, fontsize=8):
    ax.text(.08, .18, "Area Fraction (%)", transform=ax.transAxes, fontsize=fontsize)
    ax.text(.08, .11, f"Drying  {drying:.1f}", transform=ax.transAxes,
            color=(.42, .24, .02), fontsize=fontsize)
    ax.text(.08, .04, f"Wetting  {wetting:.1f}", transform=ax.transAxes,
            color=(0, .31, .25), fontsize=fontsize)


def _layout(fig, rows, cols, *, left=.045, right=.025, top=.04, bottom=.1,
            hgap=.025, vgap=.08, ordinary=()):
    width = (1 - left - right - (cols - 1) * hgap) / cols
    height = (1 - top - bottom - (rows - 1) * vgap) / rows
    axes, positions = [], []
    for k in range(rows * cols):
        row, col = divmod(k, cols)
        position = [left + col * (width + hgap),
                    1 - top - (row + 1) * height - row * vgap, width, height]
        kwargs = {} if k in ordinary else {"projection": projection()}
        axes.append(fig.add_axes(position, **kwargs))
        positions.append(position)
    return axes, positions


def _colorbar(fig, mappable, position, label, *, limits=None, fontsize=8):
    cax = fig.add_axes(position)
    bar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    if limits is not None:
        bar.set_ticks(np.linspace(-limits, limits, 5))
    bar.ax.tick_params(labelsize=fontsize, length=2)
    bar.ax.set_title(label, fontsize=fontsize, pad=6)
    return bar


def _series_matrix(path, names, years):
    mat = load_mat(path)
    values = np.asarray(find_array(mat, names), dtype=float).squeeze()
    if values.ndim != 2:
        raise ValueError(f"{path.name}: expected year-by-scheme matrix, found {values.shape}")
    if values.shape[0] != len(years) and values.shape[1] == len(years):
        values = values.T
    if values.shape[0] != len(years) or values.shape[1] < 6:
        raise ValueError(f"{path.name}: expected {len(years)} years and at least six schemes; found {values.shape}")
    return values


def _trend_series(ax, years, values, colors, names, *, stored=None, unit="z-unit decade$^{-1}$"):
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

    parts = []
    for k, (data, color, name) in enumerate(zip(values.T, colors, names)):
        if not np.isfinite(data).all():
            raise ValueError(f"{name}: annual sequence contains nonfinite values")
        input_data = np.column_stack((np.arange(1, len(years) + 1), data))
        result = ktaub(input_data, .05)
        line = source_sen_line(input_data, result)[:, 1]
        ax.plot(years, data, color=color, lw=1.2, label=name)
        ax.plot(years, line, color=color, lw=.75, ls="--")
        sen, sig = (result.sen * 10, result.h == 1) if stored is None else stored[k]
        text = f"{sen:.2f}" + ("$^{*}$" if sig else "") + (", " if k < len(names) - 1 else "")
        parts.append(TextArea(text, textprops={"color": color, "weight": "bold", "fontsize": 8}))
    parts.append(TextArea(f" ({unit})", textprops={"fontsize": 8}))
    box = AnchoredOffsetbox(loc="lower left", child=HPacker(children=parts, align="baseline", pad=0, sep=0),
                            frameon=False, bbox_to_anchor=(.01, .02), bbox_transform=ax.transAxes, borderpad=0)
    ax.add_artist(box)


def _spei_files(ctx, clim=False):
    trend_dir = ctx.path("spei_trend_dir", "SPEI/Trend/GPCC")
    index = "SPEI-PETclim12" if clim else "SPEI12"
    filenames = [
        f"HGD01-ERA5_007_{index}_GL_{YEAR_TAG}_0.25deg_trend_12thYr.mat",
        f"HGD01-ERA5_007-002_diff{index}_GL_{YEAR_TAG}_0.25deg_trend_12thYr.mat",
        f"HGD01-JRA3Q_007_{index}_GL_{YEAR_TAG}_0.25deg_trend_12thYr.mat",
        f"HGD01-JRA3Q_007-002_diff{index}_GL_{YEAR_TAG}_0.25deg_trend_12thYr.mat",
        f"HGD01-ERA5_002_{index}_GL_{YEAR_TAG}_0.25deg_trend_12thYr.mat",
    ]
    if clim:
        filenames.append(f"ERA5-JRA3Q_diff{index}_GL_{YEAR_TAG}_0.25deg_trend_12thYr.mat")
    stats = ctx.path("figure_statistics_dir", "fig/data") / f"GPCC_{index}_sig_Dr_Wet_0.25deg.mat"
    return [trend_dir / name for name in filenames], stats


def _spei_map_scale(ctx):
    unit = ctx.config.get("spei_map_input_unit", "year")
    if unit not in ("year", "decade"):
        raise ValueError("spei_map_input_unit must be 'year' or 'decade'")
    # Source Sen slopes use a one-year time step; both values and limits are
    # converted together when displaying slopes per decade.
    return (10.0 if unit == "year" else 1.0), (0.5, 0.8)


def _render_spei(ctx, clim):
    import matplotlib.pyplot as plt

    paths, area_path = _spei_files(ctx, clim)
    mask_path, mask_lon, mask_lat, mask1, _ = _mask(ctx)
    required = [mask_path, area_path, *paths]
    years = np.arange(1973, 2025)
    series = stored = None
    if not clim:
        series_dir = ctx.path("spei_series_dir", "series/GPCC")
        series_paths = [series_dir / f"HGD01-{ds}_SPEI12_GL_series_{YEAR_TAG}_12thYr.mat"
                        for ds in ("ERA5", "JRA3Q")]
        trend_paths = [series_dir / f"HGD01-{ds}_SPEI12_GL_series_{YEAR_TAG}_trend_12thYr.mat"
                       for ds in ("ERA5", "JRA3Q")]
        required += series_paths + trend_paths
    required = _require(required)
    grids = [_mat_trend(path) for path in paths]
    fractions = _vector(load_mat(area_path), ["SPEI_sig_Dr_Wet"])
    if len(fractions) < 6:
        raise ValueError("SPEI_sig_Dr_Wet requires six values in the original wet/dry ordering")
    if not clim:
        e = _series_matrix(series_paths[0], ["SPEI_series_Yr"], years)
        j = _series_matrix(series_paths[1], ["SPEI_series_Yr"], years)
        series = np.column_stack((e[:, 0], e[:, 5], j[:, 5]))
        es = load_mat(trend_paths[0])
        js = load_mat(trend_paths[1])
        e_sen = _vector(es, ["SPEI_sen_series_Yr"])
        e_flag = _vector(es, ["SPEI_p_series_Yr"])
        j_sen = _vector(js, ["SPEI_sen_series_Yr"])
        j_flag = _vector(js, ["SPEI_p_series_Yr"])
        if min(len(e_sen), len(e_flag), len(j_sen), len(j_flag)) < 6:
            raise ValueError("Stored SPEI annual trend arrays require six schemes")
        stored = [(e_sen[0] * 10, e_flag[0] == 1), (e_sen[5] * 10, e_flag[5] == 1),
                  (j_sen[5] * 10, j_flag[5] == 1)]
    if _checked(ctx, required, f"{len(grids)} map grids; six area fractions" + ("; 52 annual observations" if not clim else "")):
        return
    scale, limits = _spei_map_scale(ctx)
    term = r"SPEI$_{\mathrm{AEDclim}}$" if clim else "SPEI"
    titles = [f"ERA5 {term} Trend", f"(ERA5 − GHCD) {term} diff Trend",
              f"JRA-3Q {term} Trend", f"(JRA-3Q − GHCD) {term} diff Trend",
              f"GHCD {term} Trend", f"(ERA5 − JRA-3Q) {term} diff Trend"]
    fig = plt.figure(figsize=(14 if not clim else 12, 8.5))
    axes, positions = _layout(fig, 3, 2, top=.04, bottom=.13, hgap=.035,
                              vgap=.065, ordinary=() if clim else (5,))
    mappables = {}
    for k, (lon, lat, slope, sig) in enumerate(grids):
        shown_mask = _on_grid(mask1, mask_lon, mask_lat, lon, lat) > 0
        limit = limits[k % 2]
        mappable = map_panel(axes[k], lon, lat, slope * scale, significant=sig,
                             vmin=-limit, vmax=limit, cmap="BrBG", title=titles[k], mask=shown_mask)
        mappables[k % 2] = mappable
        if k in (0, 2, 4):
            wet_index, dry_index = {0: (2, 3), 2: (4, 5), 4: (0, 1)}[k]
            _area_text(axes[k], fractions[dry_index], fractions[wet_index])
        _label(axes[k], string.ascii_lowercase[k])
    if not clim:
        ax = axes[5]
        position = positions[5]
        ax.set_position([position[0] + position[2] * .1, position[1], position[2] * .8, position[3]])
        _trend_series(ax, years, series, COLORS, ("GHCD", "ERA5", "JRA-3Q"), stored=stored)
        ax.set(xlim=(1973, 2024), xlabel="Year", ylabel="SPEI (z-unit)", title="Global SPEI Series")
        ax.legend(frameon=False, fontsize=8, loc="best")
        _label(ax, "f")
    for col in (0, 1):
        x, _, width, _ = positions[col]
        label = f"{term} " + ("diff " if col else "") + r"Trend (z-unit decade$^{-1}$)"
        _colorbar(fig, mappables[col], [x + width * .1, .046, width * .8, .014], label,
                  limits=limits[col], fontsize=9)
    save_figure(fig, ctx)
    plt.close(fig)


def render_fig02(ctx):
    """Five SPEI trend maps and the global December SPEI-12 series."""
    _render_spei(ctx, False)


def render_figS07(ctx):
    """Six SPEI-AEDclim trend and paired-difference trend maps."""
    _render_spei(ctx, True)


def render_figS01(ctx):
    """Eighteen hydroclimatic-driver maps in six rows and three columns."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    directory = ctx.path("forcing_trend_dir", "Trend")
    fields = ("Rs", "VPD", "WS", "P", "PET", "P-PET")
    display = ("Rs", "VPD", "WS", "P", "AED", "P − AED")
    datasets = ("HGD01", "ERA5", "JRA3Q")
    display_datasets = ("GHCD", "ERA5", "JRA-3Q")
    ranges = (5, 1.2, .3, 200, 70, 220)
    radiation_palette = LinearSegmentedColormap.from_list(
        "radiation_rainbow", ["#260080", "#003aff", "#00b8ff", "#00ffd0", "#80ffb0",
                              "#ffffb0", "#ffe000", "#ff7900", "#ff0020", "#b00090"], N=256)
    palettes = (radiation_palette, "BrBG_r", "PRGn_r", "BrBG", "BrBG_r", "BrBG")
    units = (r"W m$^{-2}$ decade$^{-1}$", r"hPa decade$^{-1}$", r"m s$^{-1}$ decade$^{-1}$",
             r"mm yr$^{-1}$ decade$^{-1}$", r"mm yr$^{-1}$ decade$^{-1}$", r"mm yr$^{-1}$ decade$^{-1}$")
    paths = [directory / f"{field}_{ds}_GL_{YEAR_TAG}_0.25deg_trend_Yr.mat"
             for field in fields for ds in datasets]
    mask_path, mask_lon, mask_lat, mask1, _ = _mask(ctx)
    required = _require([mask_path, *paths])
    grids = [_mat_trend(path) for path in paths]
    if _checked(ctx, required, "18 maps; original 0/1 significance flags"):
        return
    fig = plt.figure(figsize=(14.5, 18))
    axes, positions = _layout(fig, 6, 3, right=.09, top=.025, bottom=.035,
                              hgap=.018, vgap=.035)
    for k, (lon, lat, slope, sig) in enumerate(grids):
        row, col = divmod(k, 3)
        shown_mask = _on_grid(mask1, mask_lon, mask_lat, lon, lat) > 0
        mappable = map_panel(axes[k], lon, lat, slope * 10, significant=sig,
                             vmin=-ranges[row], vmax=ranges[row], cmap=palettes[row],
                             title=f"{display_datasets[col]} {display[row]} trend", mask=shown_mask)
        _label(axes[k], string.ascii_lowercase[k], fontsize=9)
        if col == 2:
            _, y, _, height = positions[k]
            cax = fig.add_axes([.936, y, .009, height])
            bar = fig.colorbar(mappable, cax=cax)
            bar.set_label(units[row], fontsize=8)
            bar.ax.tick_params(labelsize=7)
    save_figure(fig, ctx)
    plt.close(fig)


def _rh_table(path, years):
    import pandas as pd

    table = pd.read_excel(path, sheet_name=0)
    table.columns = [str(c).strip().removesuffix(".0").casefold() for c in table.columns]
    headers = ("600", "700", "850", "925", "1000", "surface")
    if "year" not in table.columns:
        raise ValueError(f"{path.name}: missing year column")
    missing = [name for name in headers if name not in table.columns]
    if missing:
        raise ValueError(f"{path.name}: missing RH columns {missing}")
    all_years = pd.to_numeric(table["year"], errors="coerce").to_numpy()
    keep = (all_years >= years[0]) & (all_years <= years[-1])
    selected_years = all_years[keep]
    if not np.array_equal(selected_years, years):
        raise ValueError(f"{path.name}: years must be complete and ordered from {years[0]} to {years[-1]}")
    values = table.loc[keep, list(headers)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: missing or nonnumeric RH values in the selected period")
    return values


def render_figS05(ctx):
    """Ocean/land RH at five pressure levels and the surface, 1973–2024."""
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

    directory = ctx.path("rh_pressure_dir", "RH")
    paths = [directory / f"{ds}_RH_Yr/{ds}_RH_60S60N_{region}_by_pressure_1950_2025.xlsx"
             for region in ("ocean", "land") for ds in ("ERA5", "JRA3Q")]
    required = _require(paths)
    years = np.arange(1973, 2025)
    tables = [_rh_table(path, years) for path in paths]
    if _checked(ctx, required, "four complete 52-year tables; 600/700/850/925/1000 hPa and Surface"):
        return
    fig, axes = plt.subplots(6, 2, figsize=(10, 14), sharex=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.055, top=.96, hspace=.42, wspace=.17)
    titles = ("600 hPa", "700 hPa", "850 hPa", "925 hPa", "1000 hPa", "Surface")
    for row in range(6):
        for col, region in enumerate(("Ocean", "Land")):
            ax = axes[row, col]
            boxes = []
            for ds, name, color in zip(range(2), ("ERA5", "JRA-3Q"), COLORS[1:]):
                data = tables[col * 2 + ds][:, row]
                input_data = np.column_stack((np.arange(1, len(years) + 1), data))
                result = ktaub(input_data, .05)
                line = source_sen_line(input_data, result)[:, 1]
                ax.plot(years, data, color=color, lw=1.2, label=name)
                ax.plot(years, line, color=color, ls="--", lw=.7)
                text = f"{result.sen * 10:.2g}" + ("$^{*}$" if result.h == 1 else "") + (",  " if ds == 0 else "")
                boxes.append(TextArea(text, textprops={"color": color, "fontsize": 8, "weight": "bold"}))
            boxes.append(TextArea(r" (% decade$^{-1}$)", textprops={"fontsize": 8}))
            box = AnchoredOffsetbox(loc="upper right", child=HPacker(children=boxes, align="baseline", pad=0, sep=0),
                                    frameon=False, borderpad=.1)
            ax.add_artist(box)
            ax.set_title(f"{titles[row]} ({region})", fontsize=10, weight="normal", pad=8)
            ax.set_ylabel("RH (%)")
            ax.set_xlim(1972, 2025)
            ax.set_xticks((1973, 1980, 1990, 2000, 2010, 2024))
            ax.tick_params(direction="in", labelsize=9)
            _label(ax, string.ascii_lowercase[row * 2 + col])
            if row == 5:
                ax.set_xlabel("Year")
            if row == col == 0:
                ax.legend(frameon=False, fontsize=8, loc="lower left")
    save_figure(fig, ctx)
    plt.close(fig)


def _netcdf_trend(path):
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        lon_name = next((name for name in ("lon", "longitude") if name in ds.variables), None)
        lat_name = next((name for name in ("lat", "latitude") if name in ds.variables), None)
        if lon_name is None or lat_name is None:
            raise ValueError(f"{path.name}: missing longitude/latitude coordinates")
        lon = np.asarray(ds[lon_name][:], dtype=float).squeeze()
        lat = np.asarray(ds[lat_name][:], dtype=float).squeeze()
        if lon.ndim != 1 or lat.ndim != 1:
            raise ValueError(f"{path.name}: longitude/latitude must be one-dimensional")
        lon_dim = ds[lon_name].dimensions[0]
        lat_dim = ds[lat_name].dimensions[0]
        values = []
        for name in ("sen_slope", "mk_p"):
            if name not in ds.variables:
                raise ValueError(f"{path.name}: missing variable {name}")
            var = ds[name]
            raw = np.ma.filled(var[:], np.nan).astype(float)
            dims = [dim for dim, n in zip(var.dimensions, raw.shape) if n != 1]
            raw = raw.squeeze()
            if dims == [lat_dim, lon_dim]:
                value = raw
            elif dims == [lon_dim, lat_dim]:
                value = raw.T
            else:
                raise ValueError(f"{path.name}:{name}: expected latitude/longitude dimensions, found {var.dimensions}")
            if value.shape != (len(lat), len(lon)):
                raise ValueError(f"{path.name}:{name}: coordinate/field size mismatch")
            values.append(value)
    return lon, lat, values[0], values[1]


def render_figS10(ctx):
    """Nine precipitation, soil-moisture and VOD trend maps."""
    import matplotlib.pyplot as plt

    directory = ctx.path("hydroclimate_trend_dir", "hydroclimate_trends")
    stat_dir = ctx.path("figure_statistics_dir", "fig/data")
    stems = (
        ("GPCC_precip_raw", "GPCC", "1981_2024"),
        ("ESA_CCI_SM", "ESA_CCI_SM", "1981_2024"),
        ("ESA_CCI_SM", "ESA_CCI_SM", "1998_2021"),
        ("MSWEP_precip_raw", "MSWEP", "1981_2024"),
        ("GLEAM_SMrz", "GLEAM_SMrz", "1981_2024"),
        ("GLEAM_SMrz", "GLEAM_SMrz", "1998_2021"),
        ("GPCP_precip_raw", "GPCP", "1981_2024"),
        ("VODCA_CXKu", "VODCA_CXKu", "1988_2021"),
        ("VODCA_CXKu", "VODCA_CXKu", "1998_2021"),
    )
    paths = [directory / f"{stem}_sen_mk_{period}.nc" for stem, _, period in stems]
    area_paths = [stat_dir / f"{stem}_sig_Dr_Wet_{period}.mat" for _, stem, period in stems]
    mask_path, mask_lon, mask_lat, mask1, mask2 = _mask(ctx)
    if mask2 is None:
        raise ValueError("Fig. S10 requires mask2 in the forcing mask")
    required = _require([mask_path, *paths, *area_paths])
    grids = [_netcdf_trend(path) for path in paths]
    fractions = [_vector(load_mat(path), ["sig_Dr_Wet"]) for path in area_paths]
    if any(len(value) != 2 for value in fractions):
        raise ValueError("Each S9 sig_Dr_Wet array must be [wetting, drying]")
    if _checked(ctx, required, "nine NetCDF grids; original p-values; nine wet/dry area pairs"):
        return
    titles = ("GPCC Precipitation\n(1981–2024)", "ESA CCI Surface Soil Moisture\n(1981–2024)",
              "ESA CCI Surface Soil Moisture\n(1998–2021)", "MSWEP Precipitation\n(1981–2024)",
              "GLEAM Root-zone Soil Moisture\n(1981–2024)", "GLEAM Root-zone Soil Moisture\n(1998–2021)",
              "GPCP Precipitation\n(1981–2024)", "VODCA C/X/Ku-band VOD\n(1988–2021)",
              "VODCA C/X/Ku-band VOD\n(1998–2021)")
    factors = (1, 1e3, 1e3, 1, 1e3, 1e3, 1, 1e3, 1e3)
    limits = (10, .5, .5, 10, 1.5, 1.5, 10, 4, 4)
    labels = (r"Trend (mm yr$^{-2}$)", r"Trend ($10^{-3}$ m$^3$ m$^{-3}$ yr$^{-1}$)",
              r"Trend ($10^{-3}$ m$^3$ m$^{-3}$ yr$^{-1}$)", r"Trend (mm yr$^{-2}$)",
              r"Trend ($10^{-3}$ m$^3$ m$^{-3}$ yr$^{-1}$)", r"Trend ($10^{-3}$ m$^3$ m$^{-3}$ yr$^{-1}$)",
              r"Trend (mm yr$^{-2}$)", r"Trend ($10^{-3}$ yr$^{-1}$)", r"Trend ($10^{-3}$ yr$^{-1}$)")
    fig = plt.figure(figsize=(15, 8.8))
    axes, positions = _layout(fig, 3, 3, top=.075, bottom=.1, vgap=.14, hgap=.028)
    for k, (lon, lat, slope, p) in enumerate(grids):
        excluded = _on_grid(mask2, mask_lon, mask_lat, lon, lat) > 0
        display_mask = (_on_grid(mask1, mask_lon, mask_lat, lon, lat) > 0) | (~excluded & ~np.isfinite(slope))
        values = np.where(excluded, np.nan, slope * factors[k])
        significant = np.isfinite(p) & (p <= .05) & np.isfinite(values)
        mappable = map_panel(axes[k], lon, lat, values, significant=significant,
                             vmin=-limits[k], vmax=limits[k], cmap="BrBG", title=titles[k], mask=display_mask)
        _label(axes[k], string.ascii_lowercase[k])
        _area_text(axes[k], fractions[k][1], fractions[k][0], fontsize=7)
        x, y, width, _ = positions[k]
        _colorbar(fig, mappable, [x + width * .23, y - .055, width * .54, .01],
                  labels[k], limits=limits[k], fontsize=7)
    save_figure(fig, ctx)
    plt.close(fig)


def _station_columns(value, seen=None):
    if seen is None:
        seen = set()
    if id(value) in seen:
        return None
    seen.add(id(value))
    if isinstance(value, Mapping):
        lower = {str(name).casefold(): name for name in value}
        lon_key = next((lower[name] for name in ("lon", "longitude") if name in lower), None)
        lat_key = next((lower[name] for name in ("lat", "latitude") if name in lower), None)
        if lon_key is not None and lat_key is not None:
            return value[lon_key], value[lat_key]
        for item in value.values():
            found = _station_columns(item, seen)
            if found is not None:
                return found
    elif hasattr(value, "_fieldnames"):
        return _station_columns({name: getattr(value, name) for name in value._fieldnames}, seen)
    elif isinstance(value, np.ndarray) and value.dtype.names:
        return _station_columns({name: value[name] for name in value.dtype.names}, seen)
    elif isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        return _station_columns(value.item(), seen)
    return None


def _stations(path, expected):
    import pandas as pd

    suffix = path.suffix.casefold()
    if suffix == ".csv":
        table = pd.read_csv(path)
        columns = _station_columns({name: table[name].to_numpy() for name in table.columns})
    elif suffix in (".xlsx", ".xls"):
        table = pd.read_excel(path, sheet_name=0)
        columns = _station_columns({name: table[name].to_numpy() for name in table.columns})
    elif suffix == ".mat":
        columns = _station_columns(load_mat(path))
    else:
        raise ValueError("Station inputs must be CSV, Excel or MAT with lon/lat fields")
    if columns is None:
        raise ValueError(f"{path.name}: could not read longitude/latitude fields; use a plain numeric MAT struct or CSV table for opaque MATLAB table objects")
    lon, lat = (np.asarray(value, dtype=float).reshape(-1) for value in columns)
    if len(lon) != len(lat):
        raise ValueError(f"{path.name}: longitude/latitude lengths differ")
    valid = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[valid], lat[valid]
    if np.any(np.abs(lat) > 90) or np.any((lon < -180) | (lon > 360)):
        raise ValueError(f"{path.name}: geographic coordinates fall outside valid degree ranges")
    if expected is not None and len(lon) != int(expected):
        raise ValueError(f"{path.name}: expected {expected} stations after finite-coordinate filtering; found {len(lon)}")
    print(f"{path.name}: {len(valid)} rows, {len(lon)} finite-coordinate stations")
    return (lon + 180) % 360 - 180, lat


def render_figS12(ctx):
    """RH and wind station distributions in two Robinson map panels."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    rh_path = ctx.path("station_rh", "homo_station_GL/RH/stainfo/stainfo.mat")
    ws_path = ctx.path("station_wind", "homo_station_GL/wind/stainfo/stainfo.mat")
    required = _require([rh_path, ws_path])
    rh = _stations(rh_path, ctx.config.get("rh_station_count", 7275))
    ws = _stations(ws_path, ctx.config.get("wind_station_count", 5515))
    if _checked(ctx, required, f"{len(rh[0])} RH stations; {len(ws[0])} wind stations"):
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), subplot_kw={"projection": projection()})
    fig.subplots_adjust(left=.035, right=.98, bottom=.035, top=.95, hspace=.12)
    for k, (ax, values, color, name) in enumerate(zip(axes, (rh, ws), COLORS[1:], ("RH", "WS"))):
        lon, lat = values
        ax.set_extent((-180, 180, -60, 90), crs=ccrs.PlateCarree())
        ax.coastlines(resolution="110m", color=(.25, .25, .25), linewidth=.8)
        grid = ax.gridlines(draw_labels=True, linewidth=0)
        grid.top_labels = False
        grid.right_labels = False
        grid.xlabel_style = {"size": 9}
        grid.ylabel_style = {"size": 9}
        ax.scatter(lon, lat, s=3, marker="o", c=[color], alpha=.5, linewidths=.6,
                   transform=ccrs.PlateCarree(), rasterized=True)
        ax.set_title(f"{name} Stations (n = {len(lon):,})", fontsize=11, weight="normal")
        _label(ax, string.ascii_lowercase[k])
    save_figure(fig, ctx)
    plt.close(fig)
