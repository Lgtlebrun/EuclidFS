from EuclidFS.histogram import Hist1D, Hist2D
from EuclidFS.colour_cuts import apply_lrg_cuts
import polars as pl
import numpy as np
from EuclidFS.data import random_sample
from EuclidFS.config import RunDir
from pathlib import Path
import argparse

"""Computes 2D histogram of magnitude vs redshift, with given constraints, e.g. apply lrg cuts"""

if __name__ == "__main__":

    run = RunDir("mag_vs_z")

    parser = argparse.ArgumentParser()
    parser.add_argument("--redshift-max", type=float, default=3.0,
                        help="Maximum redshift cut")
    parser.add_argument("--mag-max",      type=float, default=30,
                        help="Absolute magnitude cut on W! (e.g. 0)")
    args = parser.parse_args()

    redshift_max = args.redshift_max
    mag_max      = args.mag_max

    h = Hist2D(
        x         = pl.col("true_redshift_gal"),
        y         = pl.col("wise_w1_mag"),
        x_bins    = np.linspace(0, redshift_max, 201),
        y_bins    = np.linspace(14, mag_max, 201),
        x_label   = "redshift",
        y_label   = "W1",
        prepare_fn=apply_lrg_cuts
    ).compute()

    fig, ax = h.plot()
    run.save_plot(fig, "hist2d_mag_vs_z")

    h.save(run, name="h2D_mag_vs_z")

    print("Done!")