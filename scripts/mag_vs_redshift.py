from EuclidFS.histogram import Hist2D
from EuclidFS.colour_cuts import apply_lrg_cuts
import polars as pl
import numpy as np
from EuclidFS.config import RunDir
from pathlib import Path
import argparse

"""Computes 2D histogram of magnitude vs redshift, with given constraints, e.g. apply lrg cuts"""

nocut = (None, None)
lrg_cut = ("LRG cut", apply_lrg_cuts)

if __name__ == "__main__":

    redshift_col = "true_redshift_gal"
    mag_col = "wise_w1_mag"

    parser = argparse.ArgumentParser()
    parser.add_argument("--redshift-max", type=float, default=3.0,
                        help="Maximum redshift cut")
    parser.add_argument("--mag-max",      type=float, default=30,
                        help="Absolute magnitude cut on W! (e.g. 0)")
    parser.add_argument("--mag-col", type=str, default=mag_col, 
                        help="column for magnitude")
    args = parser.parse_args()

    redshift_max = args.redshift_max
    mag_max      = args.mag_max
    mag_col = args.mag_col


    run = RunDir(f"{mag_col}_vs_z")

    cut = lrg_cut

    h = Hist2D(
        x         = pl.col(redshift_col),
        y         = pl.col(mag_col),
        x_bins    = np.linspace(0, redshift_max, 201),
        y_bins    = np.linspace(14, mag_max, 201),
        x_label   = "redshift",
        y_label   = mag_col,
        prepare_fn=cut[1]
    ).compute()

    mag_max_str = str(mag_max).replace('.', 'p')
    redshift_max_str = str(redshift_max).replace('.', 'p')

    fig, ax = h.plot(title=cut[0])
    run.save_plot(fig, f"lrg_hist2d_{mag_col}_vs_z_M{mag_max_str}_Z{redshift_max_str}")

    h.save(run, name=f"lrg_h2D_{mag_col}_vs_z_M{mag_max_str}_Z{redshift_max_str}")
    del h

    def sample(lf:pl.LazyFrame):
        return lf.head(10000000)    # 10 million rows

    sample_cut = ("10M rows", sample)

    h2 = Hist2D(
        x         = pl.col(redshift_col),
        y         = pl.col(mag_col),
        x_bins    = np.linspace(0, redshift_max, 201),
        y_bins    = np.linspace(14, mag_max, 201),
        x_label   = "redshift",
        y_label   = "W1",
        prepare_fn= sample_cut[1]
    ).compute()


    fig2, ax2 = h2.plot(title=cut[0])
    run.save_plot(fig2, f"10M_hist2d_{mag_col}_vs_z_M{mag_max_str}_Z{redshift_max_str}")

    h2.save(run, name=f"10M_h2D_{mag_col}_vs_z_M{mag_max_str}_Z{redshift_max_str}")
    del h2

    print("Done!")