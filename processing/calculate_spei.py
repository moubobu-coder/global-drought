"""Calculate Fisk-MLE SPEI from monthly P minus AED."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from drought.processing import (
    field_time, fit_spei, flatten_matlab, load_arrays, matching_field,
    month_index, precipitation_monthly, read_grid, read_mask, save_arrays,
    water_balance,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--balance", help="MAT/NPZ/NetCDF grid x month P_PET matrix")
    source.add_argument("--precipitation", help="Gridded monthly precipitation")
    p.add_argument("--balance-var", default="P_PET")
    p.add_argument("--p-var", default="P")
    p.add_argument("--p-units", default="mm/month", choices=["mm/month", "m/day", "mm/day", "mm/s"])
    p.add_argument("--aed", help="Gridded monthly AED")
    p.add_argument("--aed-var", default="PET")
    p.add_argument("--fixed-aed", action="store_true", help="Use calendar-month AED climatology")
    p.add_argument("--mask")
    p.add_argument("--mask-var", default="mask")
    p.add_argument("--exclude-mask", action="store_true", help="A mask value of 1 means exclude")
    p.add_argument("--layout", default="lon-lat-time", choices=["lon-lat-time", "time-lat-lon"])
    p.add_argument("--start", default="1973-01-01")
    p.add_argument("--end", default="2024-12-01")
    p.add_argument("--timescale", type=int, default=12)
    p.add_argument("--jobs", type=int, default=1)
    output = p.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", help="Explicit MAT or NPZ output")
    output.add_argument("--output-root", help="Root for Monthly/SPEI-style scale/product folders")
    p.add_argument("--dataset", default="ERA5", choices=["ERA5", "JRA3Q"])
    p.add_argument("--precipitation-tag", default="GPCC")
    p.add_argument("--product-tag", default="002")
    p.add_argument("--balance-output", help="Optional MAT/NPZ water-balance output")
    a = p.parse_args()
    time = month_index(a.start, a.end)
    extra = {}
    if a.balance:
        source_arrays = load_arrays(a.balance)
        if a.balance_var not in source_arrays:
            p.error("Input has no variable " + a.balance_var)
        balance = np.asarray(source_arrays[a.balance_var], dtype=float)
        for key in ("idx", "lat", "lon"):
            if key in source_arrays:
                extra[key] = source_arrays[key]
        if "time" in source_arrays:
            import pandas as pd
            input_dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(source_arrays["time"]).squeeze()))
            if not input_dates.equals(time):
                p.error("Input time coordinate differs from --start/--end")
    else:
        if not a.aed:
            p.error("--precipitation requires --aed")
        ref = read_grid(a.precipitation, a.p_var, a.layout)
        actual_time = field_time(ref, a.start)
        if not actual_time.equals(time):
            p.error("Precipitation months differ from --start/--end")
        precipitation = precipitation_monthly(ref.values, time, a.p_units)
        aed = matching_field(read_grid(a.aed, a.aed_var, a.layout, like=ref), ref)
        monthly = water_balance(precipitation, aed, time, a.fixed_aed)
        valid_mask = (np.abs(ref.lat[:, None]) <= 60) & np.any(np.isfinite(monthly), axis=0)
        if a.mask:
            valid_mask &= read_mask(a.mask, a.mask_var, ref, a.layout, a.exclude_mask)
        balance, idx = flatten_matlab(monthly, valid_mask)
        extra = {"idx": idx, "lat": ref.lat, "lon": ref.lon}
    if balance.ndim != 2 or balance.shape[1] != len(time):
        p.error("Water balance must be grid x month and match the requested date range")
    dates = time.strftime("%Y-%m-%d").to_numpy(dtype="U10")
    if a.balance_output:
        save_arrays(a.balance_output, {"P_PET": balance, "time": dates, **extra})
    values = fit_spei(balance, time, a.timescale, a.jobs)
    index_name = "SPEI-PETclim" if a.fixed_aed else "SPEI"
    if a.output:
        output_file = Path(a.output)
    else:
        folder = Path(a.output_root) / f"HGD01-{a.dataset}_GL" / str(a.timescale) / a.precipitation_tag
        output_file = folder / (f"HGD01-{a.dataset}_{a.product_tag}_{index_name}{a.timescale}"
                                f"_GL_{time[0].year}-{time[-1].year}_Mo.mat")
    save_arrays(output_file, {"SPEI": values, "time": dates, "timescale_months": a.timescale,
                             "distribution": "fisk", "fit_method": "MLE",
                             "spei_version": "0.8.2", **extra})


if __name__ == "__main__":
    main()
