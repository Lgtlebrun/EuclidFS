
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
    mag_col = "wise_w1_mag"
    select_cols = (
        ["ra_gal", "dec_gal", "abs_mag_r01", "euclid_nisp_h_mag", "sdss_r_mag", "sdss_g_mag", "sdss_z_mag", "sdss_i_mag", "wise_w1_mag", "wise_w2_mag", redshift_col, "random_index"]
    )
    print(f"Selecting columns: {select_cols}")

    # Collect ONCE - this is your working set
    print("Collecting base sample...")
    df_base = load_lazy(
        n_rows=n_rows, 
        select=select_cols, 
        filters=[pl.col(redshift_col) < 3.0]
    ).collect(streaming=True)

    n_base = len(df_base)
    print(f"Base (z < 3): {n_base:,}")

    # All subsequent filters on in-memory DataFrame - instant, no disk I/O
    for z_max in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        n = df_base.filter(pl.col(redshift_col) < z_max).height
        print(f"  z < {z_max}: {n:,}  ({100*n/n_base:.1f}%)")

    for mag_cut in range(14, 31):
        n = df_base.filter(pl.col(mag_col) < mag_cut).height
        print(f"  {mag_col} < {mag_cut}: {n:,}  ({100*n/n_base:.1f}%)")

    for z_max in [1.0, 1.5, 2.0, 2.5, 3]:
        for mag_cut in range(14, 31):
            n = df_base.filter(
                (pl.col(redshift_col) < z_max) & 
                (pl.col(mag_col) < mag_cut)
            ).height
            print(f"  z < {z_max}  &  {mag_col} < {mag_cut}: {n:,}  ({100*n/n_base:.1f}%)")

    # LRG cuts - apply on eager DataFrame
    n_lrg = apply_lrg_cuts(df_base.lazy()).collect().height
    print(f"  DESI LRG cut: {n_lrg:,}  ({100*n_lrg/n_base:.1f}%)")

