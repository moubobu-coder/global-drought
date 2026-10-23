"""Calculate KTAUB series/grid trends and significant area fractions."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from drought.processing import (
    annual_aggregate, field_time, grid_areas, grid_trends, matching_field,
    read_grid, read_mask, save_arrays, series_trend, trend_area_fractions,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["series", "grid", "area"])
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variable", default="value")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--layout", default="lon-lat-time", choices=["lon-lat-time", "time-lat-lon"])
    p.add_argument("--start", help="First month of monthly gridded input")
    p.add_argument("--first-year", type=int, help="First year of an already annual grid")
    p.add_argument("--method", choices=["sum", "mean", "december"], default="december")
    p.add_argument("--difference", help="Subtract this aligned annual series/grid before calculating Sen")
    p.add_argument("--difference-var")
    p.add_argument("--group-columns", nargs="*", default=["region", "dataset", "variable"])
    p.add_argument("--time-column", default="year")
    p.add_argument("--mask")
    p.add_argument("--mask-var", default="mask2")
    p.add_argument("--exclude-mask", action="store_true")
    p.add_argument("--slope-var", default="sen_slope")
    p.add_argument("--significance-var", default="mk_p")
    p.add_argument("--significance-kind", choices=["p", "h"], default="p")
    p.add_argument("--weights")
    p.add_argument("--weights-var", default="area")
    a = p.parse_args()
    if a.mode == "series":
        source = pd.read_csv(a.input)
        if a.variable not in source or a.time_column not in source:
            p.error("Series CSV lacks the time/value columns")
        groups = [c for c in a.group_columns if c in source.columns]
        if a.difference:
            right = pd.read_csv(a.difference)
            other_variable = a.difference_var or a.variable
            join = groups + [a.time_column]
            right = right[join + [other_variable]].rename(columns={other_variable: "__reference"})
            source = source.merge(right, on=join, validate="one_to_one", how="left")
            if source["__reference"].isna().any():
                p.error("Difference series lacks matching time/group records")
            source[a.variable] -= source.pop("__reference")
        source["fit_value"] = np.nan
        results = []
        iterator = source.groupby(groups, sort=False) if groups else [((), source)]
        for keys, part in iterator:
            part = part.sort_values(a.time_column)
            if part[a.time_column].duplicated().any():
                p.error("Series contains duplicate time records within a group")
            result = series_trend(part[a.time_column].to_numpy(), part[a.variable].to_numpy(), a.alpha)
            source.loc[part.index, "fit_value"] = result.pop("fit")
            if not isinstance(keys, tuple):
                keys = (keys,)
            results.append(dict(zip(groups, keys)) | result)
        output = Path(a.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output, index=False, float_format="%.15g")
        source.to_csv(output.with_name(output.stem + "_series.csv"), index=False, float_format="%.15g")
        return
    if a.mode == "grid":
        ref = read_grid(a.input, a.variable, a.layout)
        if ref.values.ndim != 3:
            p.error("Grid trend input needs a time axis")
        if a.first_year is not None:
            annual = ref.values
            years = np.arange(a.first_year, a.first_year + len(annual))
        else:
            annual, years = annual_aggregate(ref.values, field_time(ref, a.start), a.method)
        if a.difference:
            right = read_grid(a.difference, a.difference_var or a.variable, a.layout, like=ref)
            right_values = matching_field(right, ref)
            if a.first_year is None:
                right_annual, right_years = annual_aggregate(right_values, field_time(right, a.start), a.method)
                if not np.array_equal(years, right_years):
                    p.error("Difference input years do not match")
            else:
                right_annual = right_values
            if right_annual.shape != annual.shape:
                p.error("Difference input dimensions do not match")
            annual = annual - right_annual
        mask = np.any(np.isfinite(annual), axis=0) & (np.abs(ref.lat[:, None]) <= 60)
        if a.mask:
            mask &= read_mask(a.mask, a.mask_var, ref, a.layout, a.exclude_mask)
        result = grid_trends(annual, years, mask, a.alpha)
        output = Path(a.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() in {".nc", ".nc4"}:
            import xarray as xr
            ds = xr.Dataset({name: (("lat", "lon"), value) for name, value in result.items()},
                            coords={"lat": ref.lat, "lon": ref.lon},
                            attrs={"start_year": int(years[0]), "end_year": int(years[-1]),
                                   "alpha": a.alpha, "slope_time_unit": "year"})
            ds.to_netcdf(output)
        else:
            if output.suffix.lower() == ".mat":
                result = {name: value.T for name, value in result.items()}
                result["senmap"] = result["sen_slope"]
                result["pmap"] = result["significant"]
                order = "lon-lat-time"
            else:
                order = "time-lat-lon"
            save_arrays(output, {**result, "lat": ref.lat, "lon": ref.lon,
                                 "years": years, "array_order": order})
        return
    ref = read_grid(a.input, a.slope_var, a.layout)
    significance = matching_field(read_grid(a.input, a.significance_var, a.layout, like=ref), ref)
    h = significance <= a.alpha if a.significance_kind == "p" else significance == 1
    region = np.ones(ref.values.shape, dtype=bool) & (np.abs(ref.lat[:, None]) <= 60)
    if a.mask:
        region &= read_mask(a.mask, a.mask_var, ref, a.layout, a.exclude_mask)
    weights = grid_areas(ref.lat, ref.lon)
    if a.weights:
        weights = matching_field(read_grid(a.weights, a.weights_var, a.layout, like=ref), ref)
    wet, dry = trend_area_fractions(ref.values, h, weights, region)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        pd.DataFrame([{"wet_pct": wet, "dry_pct": dry}]).to_csv(output, index=False)
    else:
        save_arrays(output, {"sig_Dr_Wet": np.array([wet, dry])})


if __name__ == "__main__":
    main()
