# -*- coding: utf-8 -*-
"""Calculate Figure S10 trends from six NetCDF products.

Run: python figure_s10_all_results.py --base-dir /path/to/input
Help: python figure_s10_all_results.py --help

Sen slope: median of all valid pairwise slopes using actual calendar years.
MK: two-sided original Mann-Kendall, tie and continuity correction; no serial
correlation correction or multiple-testing correction.
Annual VOD: equal-weight mean of available monthly means.
No plotting, area fractions or spatial masks. Output includes 60S--60N grids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from numba import njit, prange, set_num_threads


LAT_MIN, LAT_MAX = -60., 60.


@dataclass(frozen=True)
class Product:
    key: str
    filename: str
    variable: str
    temporal: str
    units: str


PRODUCTS = {
    'GPCC': Product('GPCC',
                    'GPCC_v2025_025_annual_total_precip_1961_2024_from_monthly.nc',
                    'precip', 'annual', 'mm yr-2'),
    'MSWEP': Product('MSWEP',
                     'MSWEP_annual_precipitation_1979_2025.nc',
                     'precipitation', 'annual', 'mm yr-2'),
    'GPCP': Product('GPCP', 'GPCP_precip.mon.mean.nc',
                    'precip', 'monthly_rate', 'mm yr-2'),
    'ESA': Product('ESA',
                   'ESA_CCI_SM_COMBINED_v09.2_annual_mean_1979_2024.nc',
                   'sm', 'annual', 'm3 m-3 yr-1'),
    'GLEAM': Product('GLEAM',
                     'GLEAM_SMrz_annual_mean_global_1980_2024.nc',
                     'sm_annual_mean', 'annual', 'm3 m-3 yr-1'),
    'VOD': Product('VOD',
                   'VODCA_CXKu_monthly_mean_1988_2021.nc',
                   'VODCA_CXKu', 'monthly_mean', 'yr-1'),
}
# Figure S10 panels in row-major order.
PANELS = [('a', 'GPCC', 1981, 2024), ('b', 'ESA', 1981, 2024),
          ('c', 'ESA', 1998, 2021), ('d', 'MSWEP', 1981, 2024),
          ('e', 'GLEAM', 1981, 2024), ('f', 'GLEAM', 1998, 2021),
          ('g', 'GPCP', 1981, 2024), ('h', 'VOD', 1988, 2021),
          ('i', 'VOD', 1998, 2021)]


@njit
def sen_mk_pixel(y, years, minimum):
    good = np.isfinite(y)
    v, t = y[good], years[good]
    n = len(v)
    if n < minimum:
        return np.nan, np.nan, n
    slopes = np.empty(n * (n - 1) // 2, dtype=np.float64)
    k, s = 0, 0.
    for i in range(n - 1):
        for j in range(i + 1, n):
            delta = v[j] - v[i]
            slopes[k] = delta / (t[j] - t[i])
            k += 1
            s += (1. if delta > 0 else (-1. if delta < 0 else 0.))
    slope = np.median(slopes)
    ordered = np.sort(v)
    ties, run = 0., 1
    for i in range(1, n):
        if ordered[i] == ordered[i - 1]:
            run += 1
        else:
            ties += run * (run - 1.) * (2. * run + 5.)
            run = 1
    ties += run * (run - 1.) * (2. * run + 5.)
    variance = (n * (n - 1.) * (2. * n + 5.) - ties) / 18.
    if variance <= 0:
        return slope, 1., n
    z = (s - 1.) / math.sqrt(variance) if s > 0 else (
        (s + 1.) / math.sqrt(variance) if s < 0 else 0.)
    return slope, math.erfc(abs(z) / math.sqrt(2.)), n


@njit(parallel=True)
def sen_mk_block(values, years, minimum):
    _, ny, nx = values.shape
    slope = np.full((ny, nx), np.nan, dtype=np.float32)
    p = np.full((ny, nx), np.nan, dtype=np.float64)
    n = np.zeros((ny, nx), dtype=np.int16)
    for k in prange(ny * nx):
        iy, ix = k // nx, k % nx
        slope[iy, ix], p[iy, ix], n[iy, ix] = sen_mk_pixel(
            values[:, iy, ix].astype(np.float64), years, minimum)
    return slope, p, n


def canonical_coords(da):
    mapping = {k: v for k, v in [('latitude', 'lat'), ('longitude', 'lon')]
               if k in da.dims}
    da = da.rename(mapping)
    for dim in ['lat', 'lon']:
        if dim not in da.dims or da[dim].ndim != 1:
            raise ValueError(f'Expected regular one-dimensional lat/lon: {da.dims}')
    da = da.assign_coords(lon=(da.lon + 180.) % 360. - 180.)
    da = da.sortby('lon').sortby('lat').sel(lat=slice(LAT_MIN, LAT_MAX))
    for dim in ['lat', 'lon']:
        a = da[dim].values
        if len(a) < 2 or not np.all(np.diff(a) > 0):
            raise ValueError(f'Empty/duplicate/non-increasing coordinate: {dim}')
    return da


def source_array(ds, product, start, end):
    # Decode ONLY the analysis variable and its coordinates. GPCP time_bnds
    # contains invalid large fill values and must not be decoded with time.
    da = xr.decode_cf(ds[[product.variable]])[product.variable]
    da = canonical_coords(da)
    tdim = 'year' if 'year' in da.dims else 'time'
    if tdim == 'year':
        years = da.year.values
        if not np.all(np.isfinite(years)) or not np.all(years == years.astype(int)):
            raise ValueError('Invalid year coordinate')
        da = da.assign_coords(year=years.astype(int)).sortby('year')
        da = da.sel(year=slice(start, end))
        if len(np.unique(da.year)) != da.sizes['year']:
            raise ValueError('Duplicate annual timestamps')
        found = da.year.values
    else:
        da = da.sortby('time')
        da = da.isel(time=np.flatnonzero((da.time.dt.year >= start) &
                                        (da.time.dt.year <= end)))
        ym = da.time.dt.year.values * 100 + da.time.dt.month.values
        if len(np.unique(ym)) != len(ym):
            raise ValueError('Expected one observation per calendar month')
        found = np.unique(da.time.dt.year.values)
    missing = sorted(set(range(start, end + 1)) - set(found))
    if missing:
        raise ValueError(f'{product.key}: missing complete calendar years {missing}')
    return da.transpose(tdim, 'lat', 'lon')


def annualize(block, product, min_months):
    # Load a LATITUDE BLOCK before arithmetic: never load full global time cube.
    block = block.load().astype(np.float32)
    block = block.where(np.isfinite(block) & (abs(block) < 1.e6))
    if product.key in ['GPCC', 'MSWEP', 'GPCP']:
        block = block.where(block >= 0)
    if 'year' in block.dims:
        return block
    counts = block.groupby('time.year').count('time')
    if product.temporal == 'monthly_rate':
        units = str(block.attrs.get('units', '')).lower().replace(' ', '')
        if units not in ['mm/day', 'mmd-1', 'mmday-1']:
            raise ValueError(f'GPCP requires mm/day, got {units!r}')
        total = block * block.time.dt.days_in_month
        annual = total.groupby('time.year').sum('time', skipna=True, min_count=1)
    elif product.temporal == 'monthly_mean':
        annual = block.groupby('time.year').mean('time', skipna=True)
    else:
        raise ValueError(f'Unexpected monthly input for {product.key}')
    return annual.where(counts >= min_months).transpose('year', 'lat', 'lon')


def compute_panel(args, panel):
    letter, key, start, end = panel
    product = PRODUCTS[key]
    src = args.base_dir / product.filename
    stem = {'GPCC':'GPCC_precip_raw','MSWEP':'MSWEP_precip_raw','GPCP':'GPCP_precip_raw','ESA':'ESA_CCI_SM','GLEAM':'GLEAM_SMrz','VOD':'VODCA_CXKu'}[key]
    out = args.output_dir / f'{stem}_sen_mk_{start}_{end}.nc'
    stat = src.stat()
    config = dict(source=str(src.resolve()), source_bytes=stat.st_size,
                  source_mtime_ns=stat.st_mtime_ns, product=key,
                  period=[start, end], min_valid_years=args.min_years,
                  min_valid_months=args.min_months,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    signature = json.dumps(config, sort_keys=True)
    if out.exists() and not args.force:
        with xr.open_dataset(out) as previous:
            if previous.attrs.get('computation_signature') == signature:
                print(f'[{letter}] verified cache: {out.name}', flush=True)
                return out
    print(f'[{letter}] computing {key} {start}-{end}', flush=True)
    with xr.open_dataset(src, decode_times=False) as ds:
        da = source_array(ds, product, start, end)
        shape = (da.sizes['lat'], da.sizes['lon'])
        slope = np.full(shape, np.nan, np.float32)
        pval = np.full(shape, np.nan, np.float64)
        count = np.zeros(shape, np.int16)
        for lo in range(0, shape[0], args.lat_block):
            hi = min(lo + args.lat_block, shape[0])
            annual = annualize(da.isel(lat=slice(lo, hi)), product, args.min_months)
            # GPCC supplies the number of available months even in annual input.
            if key == 'GPCC' and 'monthly_count' in ds:
                mc = canonical_coords(ds.monthly_count).sel(year=slice(start, end))
                annual = annual.where(mc.isel(lat=slice(lo, hi)).load() >= args.min_months)
            slope[lo:hi], pval[lo:hi], count[lo:hi] = sen_mk_block(
                np.ascontiguousarray(annual.values),
                annual.year.values.astype(np.float64), args.min_years)
            print(f'  lat {hi}/{shape[0]}', flush=True)
        result = xr.Dataset({
            'sen_slope': (('lat', 'lon'), slope),
            'mk_p': (('lat', 'lon'), pval),
            'n_valid_years': (('lat', 'lon'), count)},
            coords={'lat': da.lat.values, 'lon': da.lon.values})
    result.sen_slope.attrs.update(units=product.units, long_name='Theil-Sen slope')
    result.mk_p.attrs['long_name'] = 'Original two-sided Mann-Kendall p-value'
    result.attrs.update(computation_signature=signature, product=key,
                        trend_period=f'{start}-{end}', input_file=str(src),
                        minimum_valid_years=args.min_years,
                        minimum_valid_months=args.min_months,
                        monthly_mean_rule='equal weight of valid monthly means',
                        mk_method='tie correction and continuity correction; no prewhitening; no FDR',
                        latitude_range='-60 to 60', analysis_type='original absolute values')
    temp = out.with_suffix('.tmp.nc')
    result.to_netcdf(temp, engine='netcdf4', encoding={
        v: {'zlib': True, 'complevel': 4} for v in result.data_vars})
    temp.replace(out)
    result.close()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('hydroclimate_trends'))
    parser.add_argument('--min-years', type=int, default=15)
    parser.add_argument('--min-months', type=int, default=1,
                        help='1 matches available-month aggregation in source; use 12 for complete years')
    parser.add_argument('--lat-block', type=int, default=40)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--force', action='store_true', help='Recompute even verified caches')
    args = parser.parse_args(argv)
    if not 2 <= args.min_years <= 24 or not 1 <= args.min_months <= 12:
        parser.error('min-years must be 2..24; min-months must be 1..12')
    if args.lat_block < 1 or args.threads < 1:
        parser.error('Block size and threads must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_num_threads(args.threads)
    # Check all six inputs before starting any heavy computation.
    for prod in PRODUCTS.values():
        if not (args.base_dir/prod.filename).is_file():
            raise FileNotFoundError(args.base_dir/prod.filename)
    files = [compute_panel(args, panel) for panel in PANELS]
    provenance = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                      python=platform.python_version(), numpy=np.__version__, xarray=xr.__version__,
                      outputs=[str(p) for p in files],
                      significance='original two-sided MK; no FDR or serial correction',
                      scope='nine trend grids; no plotting, spatial mask or area fractions')
    (args.output_dir/'run_manifest.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Done: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
