"""Prepare precipitation, net radiation, or vapour pressure deficit."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from drought.processing import (
    Grid, daily_to_monthly, field_time, matching_field, net_radiation,
    precipitation_monthly, radiation_mj_day, read_grid, relative_humidity,
    replace_precipitation, save_grid, vapour_pressure_deficit,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["precipitation", "radiation", "vpd"])
    p.add_argument("--output", required=True, help="Output MAT, NPZ, or NetCDF")
    p.add_argument("--layout", default="lon-lat-time", choices=["lon-lat-time", "time-lat-lon"])
    p.add_argument("--start", help="First input month, YYYY-MM-01")
    p.add_argument("--input", help="Background precipitation")
    p.add_argument("--variable", default="P")
    p.add_argument("--p-units", default="mm/month", choices=["mm/month", "m/day", "mm/day", "mm/s"])
    p.add_argument("--india", help="Monthly or daily India precipitation")
    p.add_argument("--india-var", default="RAINFALL")
    p.add_argument("--india-mask")
    p.add_argument("--india-mask-var", default="IMDmask")
    p.add_argument("--india-daily-start", help="First daily date; aggregate complete months")
    p.add_argument("--china", help="Monthly China precipitation")
    p.add_argument("--china-var", default="tp")
    p.add_argument("--china-mask")
    p.add_argument("--china-mask-var", default="chinamask")
    p.add_argument("--land-mask")
    p.add_argument("--land-mask-var", default="landmask")
    p.add_argument("--fallback", help="Monthly precipitation used only for missing land cells")
    p.add_argument("--fallback-var", default="P")
    p.add_argument("--rs", help="Corrected downward shortwave")
    p.add_argument("--rs-var", default="ssrd")
    p.add_argument("--rs-units", default="W/m2", choices=["W/m2", "J/m2/day", "MJ/m2/day"])
    p.add_argument("--down", help="Reanalysis downward shortwave")
    p.add_argument("--down-var", default="ssrd")
    p.add_argument("--up", help="Reanalysis upward shortwave")
    p.add_argument("--up-var", default="ssru")
    p.add_argument("--net-shortwave", help="ERA5 net shortwave, alternative to --up")
    p.add_argument("--net-sw-var", default="ssr")
    p.add_argument("--longwave", help="Net longwave, or downward longwave with --longwave-up")
    p.add_argument("--lw-var", default="str")
    p.add_argument("--longwave-up")
    p.add_argument("--lw-up-var", default="stru")
    p.add_argument("--reanalysis-units", default="J/m2/day", choices=["W/m2", "J/m2/day", "MJ/m2/day"])
    p.add_argument("--temperature")
    p.add_argument("--t-var", default="t2m")
    p.add_argument("--temperature-units", default="C", choices=["C", "K"])
    p.add_argument("--rh")
    p.add_argument("--rh-var", default="RH")
    p.add_argument("--dewpoint")
    p.add_argument("--dewpoint-var", default="d2m")
    a = p.parse_args()
    if a.mode == "precipitation":
        if not a.input:
            p.error("precipitation requires --input")
        ref = read_grid(a.input, a.variable, a.layout)
        dates = field_time(ref, a.start)
        ref = Grid(precipitation_monthly(ref.values, dates, a.p_units), ref.lat, ref.lon, dates)
        replacements = []
        for name, bounds in [("india", (66.5, 100, 6.5, 38.5)), ("china", (70, 140, 15, 55))]:
            filename = getattr(a, name)
            if not filename:
                continue
            local = read_grid(filename, getattr(a, name + "_var"), a.layout, bounds=bounds)
            if name == "india" and a.india_daily_start:
                monthly, local_dates = daily_to_monthly(local.values, a.india_daily_start)
                local = Grid(monthly, local.lat, local.lon, local_dates)
            else:
                local.time = dates if local.time is None else local.time
            mask_file = getattr(a, name + "_mask")
            local_mask = None
            if mask_file:
                mask_grid = read_grid(mask_file, getattr(a, name + "_mask_var"), a.layout, like=local)
                local_mask = matching_field(mask_grid, local) == 1
            replacements.append((local, local_mask))
        land = None
        if a.land_mask:
            land = matching_field(read_grid(a.land_mask, a.land_mask_var, a.layout, like=ref), ref) == 1
        fallback = read_grid(a.fallback, a.fallback_var, a.layout, like=ref) if a.fallback else None
        values = replace_precipitation(ref, replacements, land, fallback)
        save_grid(a.output, values, "P", ref, dates, {"units": "mm month-1"})
    elif a.mode == "radiation":
        if not all((a.rs, a.down, a.longwave)) or not (a.up or a.net_shortwave):
            p.error("radiation requires --rs --down --longwave and --up or --net-shortwave")
        ref = read_grid(a.rs, a.rs_var, a.layout)
        dates = field_time(ref, a.start)
        corrected = radiation_mj_day(ref.values, a.rs_units)
        def read(filename, variable):
            grid = read_grid(filename, variable, a.layout, like=ref)
            return radiation_mj_day(matching_field(grid, ref), a.reanalysis_units)
        down = read(a.down, a.down_var)
        up = read(a.up, a.up_var) if a.up else down - read(a.net_shortwave, a.net_sw_var)
        longwave = read(a.longwave, a.lw_var)
        if a.longwave_up:
            longwave -= read(a.longwave_up, a.lw_up_var)
        values = net_radiation(corrected, down, up, longwave)
        save_grid(a.output, values, "Rn", ref, dates, {"units": "MJ m-2 day-1"})
    else:
        if not a.temperature or not (a.rh or a.dewpoint):
            p.error("vpd requires --temperature and --rh or --dewpoint")
        ref = read_grid(a.temperature, a.t_var, a.layout)
        dates = field_time(ref, a.start)
        temperature = ref.values - (273.15 if a.temperature_units == "K" else 0)
        if a.rh:
            rh = matching_field(read_grid(a.rh, a.rh_var, a.layout, like=ref), ref)
        else:
            dewpoint = matching_field(read_grid(a.dewpoint, a.dewpoint_var, a.layout, like=ref), ref)
            dewpoint -= 273.15 if a.temperature_units == "K" else 0
            rh = relative_humidity(temperature, dewpoint)
        values = vapour_pressure_deficit(temperature, rh)
        save_grid(a.output, values, "VPD", ref, dates, {"units": "kPa"})


if __name__ == "__main__":
    main()
