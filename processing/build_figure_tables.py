"""Build the shared annual area and trend tables for Fig. 3 and Fig. S8."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from drought.processing import DROUGHT_CLASSES, build_area_tables, load_arrays


def labels(values):
    result = []
    for item in np.asarray(values).ravel():
        array = np.asarray(item).squeeze()
        if array.dtype.kind in "US" and array.ndim:
            result.append("".join(array.ravel().astype(str)))
        else:
            result.append(str(array))
    return result


def from_area_mats(root, names, first_year, last_year, scale_index):
    years = np.arange(first_year, last_year + 1)
    products = [("GHCD", "HGD01-ERA5", "002"),
                ("ERA5", "HGD01-ERA5", "007"),
                ("JRA-3Q", "HGD01-JRA3Q", "007")]
    parts = []
    desired = {"Africa", "Asia", "Europe", "North America", "Oceania", "South America"}
    if not desired.issubset(names):
        raise ValueError("Continent names must include all six plotted continents")
    for index_name, file_index in [("SPEI", "SPEI"), ("SPEI-AEDclim", "SPEI-PETclim")]:
        for dataset, product, tag in products:
            stem = f"{product}_{tag}_{file_index}-Darea"
            for domain in ("GL", "GLcont"):
                filename = Path(root) / f"{stem}_{domain}_series_{first_year}-{last_year}_12thYr.mat"
                values = np.asarray(load_arrays(filename)["Darea_series_Yr"], dtype=float)
                if domain == "GL":
                    if values.ndim != 3 or values.shape[:2] != (len(years), 4):
                        raise ValueError(f"Unexpected global area dimensions: {filename}, {values.shape}")
                    selected = values[:, :, scale_index, None]
                    regions = ["Global"]
                else:
                    if values.ndim != 4 or values.shape[:3] != (len(years), 4, len(names)):
                        raise ValueError(f"Unexpected continental area dimensions: {filename}, {values.shape}")
                    selected = values[:, :, :, scale_index]
                    regions = names
                for region_index, name in enumerate(regions):
                    if name != "Global" and name not in desired:
                        continue
                    for class_index, label in enumerate(DROUGHT_CLASSES):
                        parts.append(pd.DataFrame({"year": years, "region": name,
                                      "dataset": dataset, "index": index_name,
                                      "drought_class": label,
                                      "area_pct": selected[:, class_index, region_index]}))
    return pd.concat(parts, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--annual", nargs="+", help="CSV tables produced by summarize_regions")
    source.add_argument("--mat-root", help="Directory containing 12 final drought-area MAT files")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--continent-file", help="MAT containing continent_name")
    p.add_argument("--continent-names", nargs="+", help="Continent names in the exact MAT array order")
    p.add_argument("--start-year", type=int, default=1973)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--scale-index", type=int, default=2, help="Zero-based MAT scale axis index; 2 selects 12 months")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--require-final-shape", action="store_true", help="Check 8736 annual and 168 trend records")
    a = p.parse_args()
    if a.annual:
        annual = pd.concat([pd.read_csv(filename) for filename in a.annual], ignore_index=True)
    else:
        names = a.continent_names
        if names is None and a.continent_file:
            names = labels(load_arrays(a.continent_file)["continent_name"])
        if names is None:
            p.error("--mat-root requires --continent-file or --continent-names")
        annual = from_area_mats(a.mat_root, names, a.start_year, a.end_year, a.scale_index)
    annual, trends = build_area_tables(annual, a.alpha)
    if a.require_final_shape and (len(annual) != 8736 or len(trends) != 168):
        p.error("Final Fig3/S8 tables require 8736 annual and 168 trend records")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    annual.to_csv(output / "annual_area_long.csv", index=False, float_format="%.15g")
    trends.to_csv(output / "trend_statistics_long.csv", index=False, float_format="%.15g")


if __name__ == "__main__":
    main()
