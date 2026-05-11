
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from EuclidFS.config import RunDir
import polars as pl
from EuclidFS.colour_cuts import apply_colour_cuts
from EuclidFS.data import load_lazy
from EuclidFS.colour_cuts import DESI_LRG_CUT

run = RunDir("try_cuts")
redshift_col = "true_redshift_gal"
select_cols = (
    ["ra_gal", "dec_gal", "abs_mag_r01", "euclid_nisp_h", "lsst_r", "lsst_g", "lsst_z", "lsst_i", "sdss_r", "wise_w1", "wise_w2", redshift_col, "random_index"]
)
print(f"Selecting columns: {select_cols}")

# ── base lazy frame ───────────────────────────────────────────────────────────
lf_base : pl.LazyFrame= (
    load_lazy(bucket_ids=None, select=select_cols, filters = [(pl.col(redshift_col) < 3.0)]).with_columns([
            pl.when(pl.col("wise_w1") > 0)
              .then(-2.5 * pl.col("wise_w1").log(base=10) - 48.6)
              .otherwise(None)
              .alias("wise_w1_mag"),
            pl.when(pl.col("wise_w2") > 0)
              .then(-2.5 * pl.col("wise_w2").log(base=10) - 48.6)
              .otherwise(None)
              .alias("wise_w2_mag"),
        ])
)

# safe — only fetches schema, no data
print("Schema:", lf_base.collect_schema())

# safe — aggregates to a single integer on the worker side
n_base = lf_base.select(pl.len()).collect().item()
print(f"Base (z < 3): {n_base:,}")

# ── redshift cuts ─────────────────────────────────────────────────────────────
for z_max in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    n = lf_base.filter(pl.col(redshift_col) < z_max).select(pl.len()).collect(streaming=True).item()
    print(f"  z < {z_max}: {n:,}  ({100*n/n_base:.1f}%)")

# ── absolute magnitude cuts ───────────────────────────────────────────────────
print()
for mag_cut in [-14.0, -16.0, -20.0, -21.0, -21.5, -22.0]:
    n = lf_base.filter(pl.col("abs_mag_r01") < mag_cut).select(pl.len()).collect(streaming=True).item()
    print(f"  abs_mag_r01 < {mag_cut}: {n:,}  ({100*n/n_base:.1f}%)")

# ── combined z + mag cuts ─────────────────────────────────────────────────────
print()
for z_max in [1.0, 1.5, 2.0]:
    for mag_cut in [-14.0, -16.0, -20.0, -21.0, -21.5, -22.0]:
        n = (
            lf_base
            .filter(pl.col(redshift_col) < z_max)
            .filter(pl.col("abs_mag_r01") < mag_cut)
            .select(pl.len())
            .collect()
            .item()
        )
        print(f"  z < {z_max}  &  abs_mag < {mag_cut}: {n:,}  ({100*n/n_base:.1f}%)")

# ── colour cuts ───────────────────────────────────────────────────────────────
print()
n_lrg = lf_base.filter(DESI_LRG_CUT).select(pl.len()).collect(streaming=True).item()
print(f"  DESI LRG cut: {n_lrg:,}  ({100*n_lrg/n_base:.1f}%)")


