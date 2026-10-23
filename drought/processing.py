"""Forcing, AED, SPEI, regional summaries, and trend calculations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .io import load_mat
from .stats import ktaub, source_sen_line


@dataclass
class Grid:
    """A field in (time, latitude, longitude), or (latitude, longitude) order."""

    values: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    time: pd.DatetimeIndex | None = None


def load_arrays(path: str | Path) -> dict:
    """Read named arrays from MAT, NetCDF or NPZ files."""
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return load_mat(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as source:
            return {key: source[key] for key in source.files}
    if path.suffix.lower() in {".nc", ".nc4", ".netcdf"}:
        import xarray as xr
        with xr.open_dataset(path) as source:
            return {key: source[key].values for key in source.variables}
    raise ValueError("Array input must be MAT, NetCDF, or NPZ: " + str(path))


def _coordinates(arrays: dict, names: tuple[str, ...]) -> np.ndarray | None:
    for key in names:
        if key in arrays:
            return np.asarray(arrays[key]).squeeze().astype(float)
    return None


def read_grid(path: str | Path, variable: str, layout: str = "lon-lat-time",
              bounds: tuple[float, float, float, float] | None = None,
              like: Grid | None = None) -> Grid:
    """Read a field, preserving coordinate order and explicit MAT axis order.

    bounds are (west, east, south, north). Legacy arrays without coordinates
    use the 0.25-degree global grid, or explicitly supplied regional bounds.
    NetCDF named latitude/longitude dimensions take precedence over layout.
    """
    path = Path(path)
    if path.suffix.lower() in {".nc", ".nc4", ".netcdf"}:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            if variable not in ds:
                raise KeyError(f"{path}: missing variable {variable}")
            da = ds[variable]
            lat_name = next((d for d in da.dims if d.lower() in
                             {"lat", "latitude"}), None)
            lon_name = next((d for d in da.dims if d.lower() in
                             {"lon", "longitude"}), None)
            if lat_name and lon_name:
                other = [d for d in da.dims if d not in (lat_name, lon_name)]
                if len(other) > 1:
                    raise ValueError("Select one time/level axis before reading " + str(path))
                order = other + [lat_name, lon_name]
                values = np.asarray(da.transpose(*order).values, dtype=float)
                time = None
                if other and other[0] in ds and np.issubdtype(ds[other[0]].dtype, np.datetime64):
                    time = pd.DatetimeIndex(ds[other[0]].values)
                return Grid(values, ds[lat_name].values.astype(float),
                            ds[lon_name].values.astype(float), time)
    arrays = load_arrays(path)
    if variable not in arrays:
        raise KeyError(f"{path}: missing variable {variable}")
    values = np.asarray(arrays[variable], dtype=float)
    if values.ndim not in (2, 3):
        raise ValueError(f"{variable} must have two or three dimensions, got {values.shape}")
    stored_order = arrays.get("array_order")
    if stored_order is not None:
        layout = str(np.asarray(stored_order).squeeze())
    if layout == "lon-lat-time":
        values = values.T if values.ndim == 2 else values.transpose(2, 1, 0)
    elif layout != "time-lat-lon":
        raise ValueError("layout must be lon-lat-time or time-lat-lon")
    lat = _coordinates(arrays, ("lat", "latitude"))
    lon = _coordinates(arrays, ("lon", "longitude"))
    nlat, nlon = values.shape[-2:]
    if lat is None or lon is None:
        if like is not None:
            lat, lon = like.lat.copy(), like.lon.copy()
        elif bounds is not None:
            west, east, south, north = bounds
            lon = np.linspace(west, east, nlon)
            lat = np.linspace(north, south, nlat)
        elif (nlat, nlon) == (721, 1440):
            lat = np.arange(90, -90.001, -0.25)
            lon = np.arange(0, 360, 0.25)
        else:
            raise ValueError("Supply latitude/longitude arrays or regional bounds")
    if len(lat) != nlat or len(lon) != nlon:
        raise ValueError("Coordinate lengths do not match field dimensions")
    time = None
    if "time" in arrays:
        raw = np.asarray(arrays["time"]).squeeze()
        if raw.dtype.kind in "USM":
            try:
                time = pd.DatetimeIndex(pd.to_datetime(raw))
            except (ValueError, TypeError):
                time = None
    return Grid(values, np.asarray(lat), np.asarray(lon), time)


def matching_field(other: Grid, reference: Grid) -> np.ndarray:
    """Align equal grids by their actual coordinates, without interpolation."""
    def index(source, target, longitude=False):
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        if longitude:
            source, target = source % 360, target % 360
        lookup = {round(float(v), 7): i for i, v in enumerate(source)}
        try:
            return np.array([lookup[round(float(v), 7)] for v in target])
        except KeyError as exc:
            raise ValueError("Inputs are not on the same grid") from exc
    iy = index(other.lat, reference.lat)
    ix = index(other.lon, reference.lon, True)
    return np.take(np.take(other.values, iy, axis=-2), ix, axis=-1)


def read_mask(path: str | Path, variable: str, reference: Grid,
              layout: str = "lon-lat-time", exclude: bool = False) -> np.ndarray:
    values = matching_field(read_grid(path, variable, layout, like=reference), reference)
    if values.ndim != 2:
        raise ValueError("Mask must be two dimensional")
    valid = values == 0 if exclude else values == 1
    return valid & (np.abs(reference.lat[:, None]) <= 60)


def month_index(start: str, end: str | None = None,
                count: int | None = None) -> pd.DatetimeIndex:
    if end is not None:
        index = pd.date_range(start=start, end=end, freq="MS")
    elif count is not None:
        index = pd.date_range(start=start, periods=count, freq="MS")
    else:
        raise ValueError("Provide an end month or a month count")
    if len(index) == 0 or index[0] != pd.Timestamp(start):
        raise ValueError("Start date must be the first day of a month")
    return index


def field_time(grid: Grid, start: str | None) -> pd.DatetimeIndex:
    if grid.values.ndim != 3:
        raise ValueError("A time-dependent field is required")
    if grid.time is not None:
        if len(grid.time) != len(grid.values):
            raise ValueError("Time coordinate length does not match data")
        return grid.time
    if start is None:
        raise ValueError("Provide --start when the input has no time coordinate")
    return month_index(start, count=len(grid.values))


def save_arrays(path: str | Path, arrays: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mat":
        from scipy.io import savemat
        savemat(path, arrays, do_compression=True)
    elif path.suffix.lower() == ".npz":
        np.savez_compressed(path, **arrays)
    else:
        raise ValueError("Named-array output must have .mat or .npz extension")


def save_grid(path: str | Path, values: np.ndarray, variable: str,
              reference: Grid, time=None, attributes: dict | None = None) -> None:
    """Save canonical NPZ/NetCDF, or legacy (lon,lat,time) MAT."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    dates = None if time is None else pd.DatetimeIndex(time)
    if path.suffix.lower() in {".nc", ".nc4"}:
        import xarray as xr
        dims = ("lat", "lon") if values.ndim == 2 else ("time", "lat", "lon")
        coords = {"lat": reference.lat, "lon": reference.lon}
        if values.ndim == 3:
            coords["time"] = dates if dates is not None else np.arange(len(values))
        ds = xr.Dataset({variable: (dims, values)}, coords=coords)
        ds[variable].attrs.update(attributes or {})
        ds.to_netcdf(path)
    else:
        order = "time-lat-lon"
        if path.suffix.lower() == ".mat":
            values = values.T if values.ndim == 2 else values.transpose(2, 1, 0)
            order = "lon-lat-time"
        arrays = {variable: values, "lat": reference.lat, "lon": reference.lon,
                  "array_order": order}
        if dates is not None:
            arrays["time"] = dates.strftime("%Y-%m-%d").to_numpy(dtype="U10")
        arrays.update(attributes or {})
        save_arrays(path, arrays)


def saturation_vapour_pressure(temperature_c: np.ndarray) -> np.ndarray:
    temperature_c = np.asarray(temperature_c, dtype=float)
    return 0.6108 * np.exp(17.27 * temperature_c / (temperature_c + 237.3))


def relative_humidity(temperature_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    return 100 * (np.exp(17.67 * dewpoint_c / (243.5 + dewpoint_c)) /
                  np.exp(17.67 * temperature_c / (243.5 + temperature_c)))


def vapour_pressure_deficit(temperature_c: np.ndarray, rh_percent: np.ndarray) -> np.ndarray:
    return saturation_vapour_pressure(temperature_c) * (1 - rh_percent / 100)


def allen_aed(temperature_c: np.ndarray, net_radiation_mj: np.ndarray,
              rh_percent: np.ndarray, wind10_m_s: np.ndarray,
              pressure_kpa: np.ndarray, heat_flux_ratio: float = 0) -> np.ndarray:
    """Allen Penman–Monteith AED, in mm day-1."""
    t = np.asarray(temperature_c, dtype=float)
    u2 = np.asarray(wind10_m_s, dtype=float) * 4.87 / np.log(678 - 5.42)
    es = saturation_vapour_pressure(t)
    vpd = es * (1 - np.asarray(rh_percent, dtype=float) / 100)
    latent_heat = 2.501 - 0.002361 * t
    gamma = 1.013e-3 * np.asarray(pressure_kpa) / (0.622 * latent_heat)
    delta = 4098 * es / (t + 237.3) ** 2
    denominator = delta + gamma * (1 + 0.34 * u2)
    radiation_term = np.maximum(0, 0.408 * delta * net_radiation_mj *
                                (1 - heat_flux_ratio) / denominator)
    aerodynamic_term = gamma * vpd * 900 * u2 / (t + 273) / denominator
    return np.maximum(0, radiation_term + aerodynamic_term)


def days_per_month(time: pd.DatetimeIndex, ndim: int) -> np.ndarray:
    return np.asarray(time.days_in_month, dtype=float).reshape((-1,) + (1,) * (ndim - 1))


def monthly_climatology(values: np.ndarray, time: pd.DatetimeIndex) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[0] != len(time):
        raise ValueError("Time and array lengths differ")
    if set(time.month) != set(range(1, 13)):
        raise ValueError("All 12 calendar months are required for climatology")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack([np.nanmean(values[time.month == m], axis=0) for m in range(1, 13)])


def daily_to_monthly(values: np.ndarray, start: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    time = pd.date_range(start, periods=len(values), freq="D")
    periods = time.to_period("M")
    unique = periods.unique()
    if any(np.count_nonzero(periods == p) != p.days_in_month for p in unique):
        raise ValueError("Daily precipitation must contain complete calendar months")
    # MATLAB sum without omitnan: a missing daily value propagates to its month.
    return np.stack([np.sum(values[periods == p], axis=0) for p in unique]), unique.to_timestamp()


def precipitation_monthly(values: np.ndarray, time: pd.DatetimeIndex,
                          units: str = "mm/month") -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if units == "mm/month":
        return values.copy()
    factor = {"m/day": 1000, "mm/day": 1, "mm/s": 86400}.get(units)
    if factor is None:
        raise ValueError("Supported precipitation units: mm/month, m/day, mm/day, mm/s")
    return values * factor * days_per_month(time, values.ndim)


def replace_precipitation(background: Grid, replacements: list[tuple[Grid, np.ndarray | None]],
                           land: np.ndarray | None = None,
                           fallback: Grid | None = None) -> np.ndarray:
    """Replace finite regional values in list order (India, then China)."""
    result = background.values.copy()
    if result.ndim != 3:
        raise ValueError("Precipitation input must have time, latitude, longitude axes")
    for local, local_mask in replacements:
        if local.values.ndim != 3 or len(local.values) != len(result):
            raise ValueError("Replacement and background month counts differ")
        if background.time is not None and local.time is not None:
            if not background.time.equals(local.time):
                raise ValueError("Replacement and background dates differ")
        def indices(target, points, longitude=False):
            target = np.asarray(target)
            points = np.asarray(points)
            if longitude:
                target, points = target % 360, points % 360
            lookup = {round(float(v), 7): i for i, v in enumerate(target)}
            try:
                return np.array([lookup[round(float(v), 7)] for v in points])
            except KeyError as exc:
                raise ValueError("Replacement coordinates must coincide with background grid") from exc
        iy = indices(background.lat, local.lat)
        ix = indices(background.lon, local.lon, True)
        if local_mask is None:
            local_mask = np.ones(local.values.shape[-2:], dtype=bool)
        if local_mask.shape != local.values.shape[-2:]:
            raise ValueError("Replacement mask dimensions differ")
        block = result[:, iy[:, None], ix[None, :]]
        use = np.isfinite(local.values) & local_mask[None, :, :]
        block[use] = local.values[use]
        result[:, iy[:, None], ix[None, :]] = block
    if land is not None:
        if land.shape != result.shape[-2:]:
            raise ValueError("Land mask dimensions differ")
        result[:, ~land] = np.nan
        missing = land[None, :, :] & ~np.isfinite(result)
        if fallback is not None:
            fallback_values = matching_field(fallback, background)
            if fallback_values.shape != result.shape:
                raise ValueError("Fallback dimensions differ")
            use = missing & np.isfinite(fallback_values)
            result[use] = fallback_values[use]
            missing = land[None, :, :] & ~np.isfinite(result)
        if np.any(missing):
            raise ValueError("Land precipitation remains missing after replacement/fallback")
    return result


def radiation_mj_day(values: np.ndarray, units: str) -> np.ndarray:
    factor = {"W/m2": 86400 / 1e6, "J/m2/day": 1e-6, "MJ/m2/day": 1}.get(units)
    if factor is None:
        raise ValueError("Supported radiation units: W/m2, J/m2/day, MJ/m2/day")
    return np.asarray(values, dtype=float) * factor


def net_radiation(corrected_shortwave: np.ndarray, reanalysis_down: np.ndarray,
                   reanalysis_up: np.ndarray, net_longwave: np.ndarray) -> np.ndarray:
    """All arguments are MJ m-2 day-1; net longwave is downward positive."""
    with np.errstate(divide="ignore", invalid="ignore"):
        albedo = reanalysis_up / reanalysis_down
    bad = (albedo < 0) | (albedo > 1)
    if np.any(bad):
        raise ValueError("Reanalysis albedo contains values outside [0, 1]")
    return corrected_shortwave * (1 - albedo) + net_longwave


def water_balance(precipitation: np.ndarray, aed: np.ndarray,
                   time: pd.DatetimeIndex, fixed_aed: bool = False) -> np.ndarray:
    precipitation = np.asarray(precipitation, dtype=float)
    aed = np.asarray(aed, dtype=float)
    if fixed_aed:
        climatology = monthly_climatology(aed, time)
        aed = climatology[np.asarray(time.month) - 1]
    if aed.shape != precipitation.shape:
        raise ValueError("P and AED dimensions differ")
    return precipitation - aed


def flatten_matlab(values: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonical time/lat/lon to legacy grid/month and logical lon-first idx."""
    idx = np.asarray(valid_mask).T.reshape(-1, order="F").astype(bool)
    flat = np.asarray(values).transpose(2, 1, 0).reshape((-1, len(values)), order="F")
    return flat[idx], idx


def restore_matlab(values: np.ndarray, idx: np.ndarray, nlat: int, nlon: int) -> np.ndarray:
    values = np.asarray(values)
    idx = np.asarray(idx).squeeze().astype(bool)
    if idx.size != nlat * nlon or np.count_nonzero(idx) != values.shape[0]:
        raise ValueError("idx and selected-grid data dimensions differ")
    full = np.full((nlon * nlat, values.shape[1]), np.nan)
    full[idx] = values
    return full.reshape((nlon, nlat, values.shape[1]), order="F").transpose(2, 1, 0)


def fit_spei(balance: np.ndarray, time: pd.DatetimeIndex, timescale: int = 12,
              jobs: int = 1) -> np.ndarray:
    """Fisk MLE SPEI using spei 0.8.2, retaining the full monthly axis."""
    import spei as si
    from scipy.stats import fisk
    balance = np.asarray(balance, dtype=float)
    if balance.ndim != 2 or balance.shape[1] != len(time):
        raise ValueError("Balance must be grid x month with a matching date axis")
    if not time.equals(pd.date_range(time[0], periods=len(time), freq="MS")):
        raise ValueError("SPEI dates must be consecutive month starts")
    if not 1 <= timescale <= len(time):
        raise ValueError("Invalid SPEI timescale")
    def one(values):
        if np.all(np.isnan(values)):
            return np.full(len(time), np.nan, dtype=np.float32)
        surplus = pd.Series(values, index=time, dtype="float64")
        fitted = si.spei(series=surplus, dist=fisk, timescale=timescale,
                         fit_freq=None, fit_window=0, prob_zero=False)
        return fitted.reindex(time).to_numpy(dtype=np.float32)
    if jobs == 1:
        return np.stack([one(row) for row in balance])
    from joblib import Parallel, delayed
    return np.stack(Parallel(n_jobs=jobs)(delayed(one)(row) for row in balance))


def annual_aggregate(values: np.ndarray, time: pd.DatetimeIndex,
                      method: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if len(values) != len(time):
        raise ValueError("Time and array lengths differ")
    years = np.unique(time.year)
    result = []
    for year in years:
        take = time.year == year
        months = np.asarray(time.month[take])
        if len(months) != 12 or set(months) != set(range(1, 13)):
            raise ValueError(f"Year {year} does not contain 12 distinct months")
        block = values[take]
        if method == "sum":
            result.append(np.sum(block, axis=0))
        elif method == "mean":
            result.append(np.mean(block, axis=0))
        elif method == "december":
            result.append(block[months == 12][0])
        else:
            raise ValueError("Annual method must be sum, mean, or december")
    return np.stack(result), years


def grid_areas(lat: np.ndarray, lon: np.ndarray,
                radius: float = 6371220) -> np.ndarray:
    """Cosine-latitude grid-cell areas in m2, preserving the latitude order."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) < 2 or len(lon) < 2:
        raise ValueError("At least two latitude and longitude centres are required")
    radians = np.pi / 180
    rr = radius * radians
    dx = abs(lon[1] - lon[0]) * rr * np.cos(lat * radians)
    dy = np.empty(len(lat), dtype=float)
    dy[0] = abs(lat[1] - lat[0]) * rr
    dy[-1] = abs(lat[-1] - lat[-2]) * rr
    dy[1:-1] = np.abs(lat[2:] - lat[:-2]) * rr * 0.5
    return np.broadcast_to((dx * dy)[:, None], (len(lat), len(lon))).copy()


def weighted_mean(values: np.ndarray, weights: np.ndarray,
                   region: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape[-2:] != weights.shape or region.shape != weights.shape:
        raise ValueError("Weights/region dimensions differ from field")
    valid = np.isfinite(values) & region & np.isfinite(weights) & (weights > 0)
    denominator = np.sum(np.where(valid, weights, 0), axis=(-2, -1))
    numerator = np.sum(np.where(valid, values * weights, 0), axis=(-2, -1))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


DROUGHT_CLASSES = ("All", "Moderate", "Severe", "Extreme")


def drought_area(values: np.ndarray, weights: np.ndarray,
                  region: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & region & np.isfinite(weights) & (weights > 0)
    denominator = np.sum(np.where(valid, weights, 0), axis=(-2, -1))
    conditions = (values < -1, (values >= -1.5) & (values < -1),
                  (values >= -2) & (values < -1.5), values < -2)
    fractions = []
    for condition in conditions:
        numerator = np.sum(np.where(valid & condition, weights, 0), axis=(-2, -1))
        fractions.append(np.divide(100 * numerator, denominator,
                         out=np.full_like(numerator, np.nan), where=denominator > 0))
    return np.stack(fractions, axis=-1)


def regional_table(annual: np.ndarray, years: np.ndarray, lat: np.ndarray,
                    lon: np.ndarray, regions: dict[str, np.ndarray], dataset: str,
                    variable: str, unit: str = "", drought: bool = False,
                    weights: np.ndarray | None = None) -> pd.DataFrame:
    if weights is None:
        weights = grid_areas(lat, lon)
    rows = []
    for name, mask in regions.items():
        if drought:
            values = drought_area(annual, weights, mask)
            for column, label in enumerate(DROUGHT_CLASSES):
                rows.extend({"year": int(y), "region": name, "dataset": dataset,
                             "index": variable, "drought_class": label,
                             "area_pct": float(value)}
                            for y, value in zip(years, values[:, column]))
        else:
            values = weighted_mean(annual, weights, mask)
            rows.extend({"year": int(y), "region": name, "dataset": dataset,
                         "variable": variable, "value": float(value), "unit": unit}
                        for y, value in zip(years, values))
    return pd.DataFrame(rows)


def series_trend(time: np.ndarray, values: np.ndarray, alpha: float = 0.05) -> dict:
    time, values = np.asarray(time, dtype=float), np.asarray(values, dtype=float)
    valid = np.isfinite(time) & np.isfinite(values)
    line = np.full(values.shape, np.nan)
    if np.count_nonzero(valid) < 3:
        return {"sen_slope": np.nan, "mk_p": np.nan, "significant": False,
                "fit": line, "n": int(np.count_nonzero(valid))}
    data = np.column_stack((time[valid], values[valid]))
    result = ktaub(data, alpha)
    line[valid] = source_sen_line(data, result)[:, 1]
    return {"sen_slope": result.sen, "mk_p": result.p,
            "significant": bool(result.h), "fit": line, "n": result.n}


def grid_trends(annual: np.ndarray, years: np.ndarray, valid_mask: np.ndarray,
                 alpha: float = 0.05) -> dict[str, np.ndarray]:
    annual = np.asarray(annual, dtype=float)
    if annual.ndim != 3 or annual.shape[1:] != valid_mask.shape or len(years) != len(annual):
        raise ValueError("Expected annual time/lat/lon field and matching valid mask")
    slope = np.full(valid_mask.shape, np.nan)
    p = np.full(valid_mask.shape, np.nan)
    significant = np.zeros(valid_mask.shape, dtype=np.uint8)
    n = np.zeros(valid_mask.shape, dtype=np.uint16)
    for iy, ix in zip(*np.where(valid_mask)):
        result = series_trend(years, annual[:, iy, ix], alpha)
        slope[iy, ix], p[iy, ix] = result["sen_slope"], result["mk_p"]
        significant[iy, ix], n[iy, ix] = result["significant"], result["n"]
    return {"sen_slope": slope, "mk_p": p, "significant": significant, "n_valid": n}


def trend_area_fractions(slope: np.ndarray, significant: np.ndarray,
                           weights: np.ndarray, region: np.ndarray) -> np.ndarray:
    valid = np.isfinite(slope) & region & np.isfinite(weights) & (weights > 0)
    total = np.sum(weights[valid])
    if total <= 0:
        return np.array([np.nan, np.nan])
    sig = np.asarray(significant, dtype=bool)
    return 100 / total * np.array([np.sum(weights[valid & sig & (slope > 0)]),
                                   np.sum(weights[valid & sig & (slope < 0)])])


def build_area_tables(annual: pd.DataFrame, alpha: float = 0.05
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["region", "dataset", "index", "drought_class"]
    required = keys + ["year", "area_pct"]
    missing = set(required) - set(annual.columns)
    if missing:
        raise ValueError("Missing annual-area columns: " + ", ".join(sorted(missing)))
    annual = annual.copy()
    annual["index"] = annual["index"].replace({"SPEI-PETclim": "SPEI-AEDclim"})
    if annual.duplicated(keys + ["year"]).any():
        raise ValueError("Duplicate annual area records")
    closure = annual.pivot(index=["region", "dataset", "index", "year"],
                           columns="drought_class", values="area_pct")
    if all(label in closure for label in DROUGHT_CLASSES):
        comparable = closure.dropna(subset=list(DROUGHT_CLASSES))
        difference = comparable["All"] - comparable[list(DROUGHT_CLASSES[1:])].sum(axis=1)
        if np.any(np.abs(difference) > 1e-8):
            raise ValueError("Drought classes do not sum to All")
    annual["fit_value_pct"] = np.nan
    trends = []
    for group, part in annual.groupby(keys, sort=False):
        part = part.sort_values("year")
        result = series_trend(part.year.to_numpy(), part.area_pct.to_numpy(), alpha)
        annual.loc[part.index, "fit_value_pct"] = result["fit"]
        trends.append(dict(zip(keys, group)) | {
            "trend_pp_decade": result["sen_slope"] * 10, "mk_p": result["mk_p"],
            "significant": result["significant"], "start_year": int(part.year.min()),
            "end_year": int(part.year.max())})
    return annual.sort_values(keys + ["year"]), pd.DataFrame(trends)
