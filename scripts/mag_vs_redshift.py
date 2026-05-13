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

    run = RunDir("mag_vs_z")
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

    cut = lrg_cut

    h = Hist2D(
        x         = pl.col(redshift_col),
        y         = pl.col(mag_col),
        x_bins    = np.linspace(0, redshift_max, 201),
        y_bins    = np.linspace(14, mag_max, 201),
        x_label   = "redshift",
        y_label   = "W1",
        prepare_fn=cut[1]
    ).compute()

    fig, ax = h.plot(title=cut[0])
    run.save_plot(fig, f"hist2d_{mag_col}_vs_z_M{mag_max}_Z{redshift_max}")

    h.save(run, name=f"h2D_{mag_col}_vs_z_M{mag_max}_Z{redshift_max}")

    print("Done!")