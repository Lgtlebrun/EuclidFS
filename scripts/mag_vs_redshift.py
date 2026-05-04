import argparse
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from EuclidFS.data import load_lazy
from EuclidFS.config import RunDir

PARAMS = {
    "mag_cols"    : ["abs_mag_r01", "euclid_nisp_h", "lsst_r", "lsst_i", "sdss_r"],
    "redshift_col": "true_redshift_gal",
    "sample_frac" : 0.001,
    "bucket_ids"  : None,
}

def plot_mag_vs_redshift(z, mag, mag_col, redshift_col, redshift_max, mag_max):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h = axes[0].hist2d(z, mag, bins=200, cmap="inferno")
    fig.colorbar(h[3], ax=axes[0], label="count")
    axes[0].set_xlabel(redshift_col)
    axes[0].set_ylabel(mag_col)
    axes[0].set_title("2D histogram")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(z), min(50_000, len(z)), replace=False)
    axes[1].scatter(z[idx], mag[idx], s=0.3, alpha=0.3, c="steelblue")
    axes[1].set_xlabel(redshift_col)
    axes[1].set_ylabel(mag_col)
    axes[1].set_title(f"Scatter (N={len(idx):,})")
    cut_info = f"z < {redshift_max}"
    if mag_max is not None:
        cut_info += f"  |  abs_mag_r01 < {mag_max}"
    plt.suptitle(f"{mag_col} vs {redshift_col}  [{cut_info}]")
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redshift-max", type=float, default=3.0,
                        help="Maximum redshift cut")
    parser.add_argument("--mag-max",      type=float, default=None,
                        help="Absolute magnitude cut on abs_mag_r01 (e.g. -21.0)")
    args = parser.parse_args()

    redshift_max = args.redshift_max
    mag_max      = args.mag_max

    # encode cuts in run name so each subjob gets its own output dir
    cut_tag = f"z{redshift_max}"
    if mag_max is not None:
        cut_tag += f"_m{mag_max}"
    run = RunDir(f"mag_vs_redshift_{cut_tag}")

    params_out = {**PARAMS, "redshift_max": redshift_max, "mag_max": mag_max}
    run.save_result(params_out, "params.json")

    select_cols = (
        ["galaxy_id", PARAMS["redshift_col"], "random_index"]
        + PARAMS["mag_cols"]
    )

    print(f"Loading data lazily  (z < {redshift_max}, mag_max={mag_max})...")
    lf = (
        load_lazy(bucket_ids=PARAMS["bucket_ids"], select=select_cols)
        .filter(pl.col(PARAMS["redshift_col"]) < redshift_max)
        .filter(pl.col("random_index") < PARAMS["sample_frac"])
    )
    if mag_max is not None:
        lf = lf.filter(pl.col("abs_mag_r01") < mag_max)

    print(lf.explain(optimized=True))
    df = lf.collect()
    print(f"Got {len(df):,} galaxies after cuts, plotting...")

    if len(df) == 0:
        print("WARNING: no galaxies pass cuts — skipping plots.")
        sys.exit(0)

    z = df[PARAMS["redshift_col"]].to_numpy()
    for mag_col in PARAMS["mag_cols"]:
        print(f"  plotting {mag_col}...")
        mag = df[mag_col].to_numpy()
        fig = plot_mag_vs_redshift(z, mag, mag_col, PARAMS["redshift_col"],
                                   redshift_max, mag_max)
        run.save_plot(fig, f"{mag_col}_vs_{PARAMS['redshift_col']}.png")
        plt.close(fig)

    run.save_result(
        df.select(PARAMS["mag_cols"] + [PARAMS["redshift_col"]]).describe(),
        "summary_stats.csv"
    )
    print("Done!")