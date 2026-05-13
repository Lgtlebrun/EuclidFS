
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from EuclidFS.config import RunDir
import polars as pl
from EuclidFS.colour_cuts import apply_lrg_cuts
from EuclidFS.data import load_lazy

import argparse

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=int(1e6), help="Number of random rows")
    parser.add_argument("--redshift-max", type=float, default=3.0,
                        help="Maximum redshift cut")
    parser.add_argument("--mag-max",      type=float, default=30,
                        help="Absolute magnitude cut on W! (e.g. 0)")
    args = parser.parse_args()

    redshift_max = args.redshift_max
    mag_max      = args.mag_max
    n_rows = args.n_rows

    run = RunDir("try_cuts")
    redshift_col = "true_redshift_gal"
    select_cols = (
        ["ra_gal", "dec_gal", "abs_mag_r01", "euclid_nisp_h_mag", "lsst_r_mag", "lsst_g_mag", "lsst_z_mag", "lsst_i_mag", "sdss_r_mag", "wise_w1_mag", "wise_w2_mag", redshift_col, "random_index"]
    )
    print(f"Selecting columns: {select_cols}")

    # ── base lazy frame ───────────────────────────────────────────────────────────
    lf_base : pl.LazyFrame=load_lazy(n_rows=n_rows, select=select_cols, filters = [(pl.col(redshift_col) < 3.0)])


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
    n_lrg = lf_base.pipe(apply_lrg_cuts).select(pl.len()).collect(streaming=True).item()
    print(f"  DESI LRG cut: {n_lrg:,}  ({100*n_lrg/n_base:.1f}%)")


