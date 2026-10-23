"""Calculate monthly Allen AED or its calendar-month climatology."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drought.processing import (
    allen_aed, days_per_month, field_time, matching_field, monthly_climatology,
    radiation_mj_day, read_grid, relative_humidity, save_grid,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["calculate", "climatology"])
    p.add_argument("--output", required=True)
    p.add_argument("--layout", default="lon-lat-time", choices=["lon-lat-time", "time-lat-lon"])
    p.add_argument("--start", help="First input month, YYYY-MM-01")
    p.add_argument("--input", help="Monthly AED for climatology")
    p.add_argument("--aed-var", default="PET")
    p.add_argument("--temperature")
    p.add_argument("--t-var", default="t2m")
    p.add_argument("--temperature-units", default="C", choices=["C", "K"])
    p.add_argument("--rh")
    p.add_argument("--rh-var", default="RH")
    p.add_argument("--dewpoint")
    p.add_argument("--dewpoint-var", default="d2m")
    p.add_argument("--wind")
    p.add_argument("--wind-var", default="WS")
    p.add_argument("--pressure")
    p.add_argument("--pressure-var", default="sp")
    p.add_argument("--pressure-units", default="kPa", choices=["kPa", "Pa"])
    p.add_argument("--radiation")
    p.add_argument("--rn-var", default="Rn")
    p.add_argument("--radiation-units", default="MJ/m2/day", choices=["W/m2", "J/m2/day", "MJ/m2/day"])
    p.add_argument("--climatology-start", type=int, default=1973)
    p.add_argument("--climatology-end", type=int, default=2024)
    a = p.parse_args()
    if a.mode == "climatology":
        if not a.input:
            p.error("climatology requires --input")
        ref = read_grid(a.input, a.aed_var, a.layout)
        time = field_time(ref, a.start)
        select = (time.year >= a.climatology_start) & (time.year <= a.climatology_end)
        if not select.any():
            p.error("No months fall within the climatology period")
        values = monthly_climatology(ref.values[select], time[select])
        from drought.processing import month_index
        save_grid(a.output, values, "PET_clim", ref, month_index("2001-01-01", count=12),
                  {"units": "mm month-1", "climatology_start": a.climatology_start,
                   "climatology_end": a.climatology_end})
    else:
        if not all((a.temperature, a.wind, a.pressure, a.radiation)) or not (a.rh or a.dewpoint):
            p.error("calculate requires --temperature --wind --pressure --radiation and --rh or --dewpoint")
        ref = read_grid(a.temperature, a.t_var, a.layout)
        time = field_time(ref, a.start)
        temperature = ref.values - (273.15 if a.temperature_units == "K" else 0)
        def read(filename, variable):
            return matching_field(read_grid(filename, variable, a.layout, like=ref), ref)
        rh = read(a.rh, a.rh_var) if a.rh else relative_humidity(
            temperature, read(a.dewpoint, a.dewpoint_var) - (273.15 if a.temperature_units == "K" else 0))
        wind = read(a.wind, a.wind_var)
        pressure = read(a.pressure, a.pressure_var) / (1000 if a.pressure_units == "Pa" else 1)
        radiation = radiation_mj_day(read(a.radiation, a.rn_var), a.radiation_units)
        values = allen_aed(temperature, radiation, rh, wind, pressure) * days_per_month(time, 3)
        save_grid(a.output, values, "PET", ref, time, {"units": "mm month-1"})


if __name__ == "__main__":
    main()
