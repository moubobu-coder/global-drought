"""Summarize annual forcing, December SPEI, or drought area by region."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from drought.processing import (
    Grid, annual_aggregate, field_time, load_arrays, matching_field,
    month_index, read_grid, read_mask, regional_table, restore_matlab,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--variable", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True, help="CSV annual regional table")
    p.add_argument("--start", help="First input month for arrays without dates")
    p.add_argument("--layout", default="lon-lat-time", choices=["lon-lat-time", "time-lat-lon"])
    p.add_argument("--method", choices=["sum", "mean", "december"], default="mean")
    p.add_argument("--drought-area", action="store_true")
    p.add_argument("--index-name", default="SPEI")
    p.add_argument("--unit", default="")
    p.add_argument("--mask")
    p.add_argument("--mask-var", default="mask2")
    p.add_argument("--exclude-mask", action="store_true")
    p.add_argument("--continent-mask")
    p.add_argument("--continent-var", default="continent_mask")
    p.add_argument("--regions", nargs="*", help="Code:name pairs, e.g. 1:Africa 2:Asia")
    p.add_argument("--weights", help="Optional external area weights")
    p.add_argument("--weights-var", default="area")
    a = p.parse_args()
    arrays = load_arrays(a.input)
    values = np.asarray(arrays.get(a.variable))
    if values.ndim == 2 and "idx" in arrays:
        if "lat" in arrays and "lon" in arrays:
            lat, lon = np.asarray(arrays["lat"]).squeeze(), np.asarray(arrays["lon"]).squeeze()
        elif np.asarray(arrays["idx"]).size == 1440 * 721:
            lat, lon = np.arange(90, -90.001, -0.25), np.arange(0, 360, 0.25)
        else:
            p.error("Compressed grid input requires latitude/longitude arrays")
        monthly = restore_matlab(values, arrays["idx"], len(lat), len(lon))
        ref = Grid(monthly, lat, lon)
        if "time" in arrays:
            import pandas as pd
            ref.time = pd.DatetimeIndex(pd.to_datetime(np.asarray(arrays["time"]).squeeze()))
        else:
            if not a.start:
                p.error("Compressed monthly input requires --start or a time coordinate")
            ref.time = month_index(a.start, count=values.shape[1])
    else:
        ref = read_grid(a.input, a.variable, a.layout)
    time = field_time(ref, a.start)
    method = "december" if a.drought_area else a.method
    annual, years = annual_aggregate(ref.values, time, method)
    valid = np.ones(annual.shape[-2:], dtype=bool) & (np.abs(ref.lat[:, None]) <= 60)
    if a.mask:
        valid &= read_mask(a.mask, a.mask_var, ref, a.layout, a.exclude_mask)
    regions = {"Global": valid}
    if a.continent_mask:
        mask_grid = read_grid(a.continent_mask, a.continent_var, a.layout, like=ref)
        codes = matching_field(mask_grid, ref)
        if not a.regions:
            p.error("--continent-mask requires explicit --regions Code:Name mappings")
        for specification in a.regions:
            code, name = specification.split(":", 1)
            regions[name] = valid & (codes == int(code))
    weights = None
    if a.weights:
        weights = matching_field(read_grid(a.weights, a.weights_var, a.layout, like=ref), ref)
    variable = a.index_name if a.drought_area else a.variable
    table = regional_table(annual, years, ref.lat, ref.lon, regions, a.dataset,
                           variable, a.unit, a.drought_area, weights)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.output, index=False, float_format="%.15g")


if __name__ == "__main__":
    main()
