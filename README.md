# Global drought analysis and plotting code

Figure numbering follows the manuscript: Figs. 1–4 and S1–S12.

Python 3.12. Install and run from this directory:

```sh
python -m pip install -r requirements.txt
python figures/fig01.py --input-root /path/to/data --output-dir figure_output
```

`figures/fig01.py`–`fig04.py` correspond to the four main figures; `figures/figS01.py`–`figS12.py` correspond to the supplementary figures. Fig. 4 compares SSI and SPEI; S9 shows ENSO correlations; S10 shows precipitation, soil-moisture and VOD trends; S11 shows regional precipitation; S12 shows station networks.

Use `--help` for options and `--check-inputs` to check inputs without drawing. A JSON file passed with `--config` can override locations through its `paths` object, for example `{"paths": {"source_data": "/path/to/source_data"}}`.

S11 reads `regional_precipitation.csv` with columns `region,year,GHCD,GPCC,CRU`. Regions are `China`, `Africa`, `Africa_no_support`, each covering 1973–2024; precipitation is in mm/year. Its path can be overridden with the `regional_precipitation` key.

`processing/` contains forcing, AED, SPEI, regional summary, trend and figure-table programs. `figure_s10_all_results.py` calculates the nine S10 trend grids from six NetCDF products:

```sh
python processing/figure_s10_all_results.py --base-dir /path/to/input --output-dir /path/to/data/hydroclimate_trends
```

Shared functions are in `drought/`. Run each processing program with `--help` for input variables and output options.## LicenseThe project code is licensed under MIT, except for the KTAUB implementationin `drought/stats.py`, which retains its BSD 3-Clause license in`KTAUB_LICENSE.txt`.
