import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from EuclidFS.data import load_lazy
from EuclidFS.config import RunDir
from EuclidFS.colour_cuts import apply_colour_cuts, DESI_LRG_CUT, DESI_IS_NORTH, DESI_IS_SOUTH, DESI_CUTS_N, DESI_CUTS_S

PARAMS = {
    "mag_cols"     : ["abs_mag_r01", "euclid_nisp_h", "lsst_r", "lsst_i", "sdss_r", "wise_w1_mag"],
    "redshift_col" : "true_redshift_gal",
    "redshift_max" : 3.0,
    "sample_frac"  : 0.001,
    "bucket_ids"   : None,
}

CUT_COLS = ["lsst_g", "lsst_r", "lsst_z", "wise_w1_mag", "dec_gal"]


# ── plot helpers ──────────────────────────────────────────────────────────────

def plot_mag_vs_redshift(z, mag, mag_col, redshift_col):
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
    plt.suptitle(f"{mag_col} vs {redshift_col}")
    plt.tight_layout()
    return fig


def plot_colour_plane(x_vals, y_vals, xlabel, ylabel, cut_fn=None, title=None, n_scatter=50_000):
    fig, ax = plt.subplots(figsize=(7, 6))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_vals), min(n_scatter, len(x_vals)), replace=False)
    ax.scatter(x_vals[idx], y_vals[idx], s=0.3, alpha=0.3, c="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{ylabel} vs {xlabel}")
    if cut_fn is not None:
        x_min, x_max = np.nanpercentile(x_vals, [1, 99])
        x_line = np.linspace(x_min, x_max, 500)
        ax.plot(x_line, cut_fn(x_line), color="red", lw=1.5, label="cut boundary")
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def _cut_zw1_rz(rz):
    return 0.8 * rz - 0.6

def _cut_rw1_w1_north(w1):
    return 1.8 * (w1 - 17.14)

def _cut_rw1_w1_south(w1):
    return 1.83 * (w1 - 17.13)


if __name__ == "__main__":
    run = RunDir("lrg_selection")
    run.save_result(PARAMS, "params.json")

    select_cols = (
        ["galaxy_id", PARAMS["redshift_col"], "random_index"]
        + [c for c in PARAMS["mag_cols"] if c not in ("wise_w1_mag", "wise_w2_mag")]
        + [c for c in CUT_COLS if c not in PARAMS["mag_cols"]]
        + ["wise_w1", "wise_w2"]
    )
    print(f"Selecting columns: {select_cols}")

    # single lazy frame — reused everywhere
    lf_base = (
        load_lazy(bucket_ids=PARAMS["bucket_ids"], select=select_cols)
        .filter(pl.col(PARAMS["redshift_col"]) < PARAMS["redshift_max"])
        .filter(pl.col("random_index") < PARAMS["sample_frac"])
        .with_columns([
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

    # ── single cheap diagnostic collect ──────────────────────────────────────
    print("--- DIAGNOSTIC (limit 1000) ---")
    df_diag = lf_base.limit(1000).collect()
    print(f"Schema: {df_diag.schema}")
    print(f"Rows after sample_frac filter (limit 1000): {len(df_diag)}")
    print(df_diag.select(["wise_w1", "wise_w1_mag", "lsst_z", "lsst_r", "dec_gal"]).head(5))
    print(f"wise_w1_mag stats:\n{df_diag['wise_w1_mag'].describe()}")
    print(f"random_index range: {df_diag['random_index'].min():.6f} – {df_diag['random_index'].max():.6f}")

    # check each cut on the 1000-row sample
    print("\n--- CUT DIAGNOSTICS (limit 1000) ---")
    print(f"  total:         {len(df_diag)}")
    print(f"  IS_NORTH:      {df_diag.filter(DESI_IS_NORTH).height}")
    print(f"  IS_SOUTH:      {df_diag.filter(DESI_IS_SOUTH).height}")
    for i, expr in enumerate(DESI_CUTS_N):
        print(f"  NORTH cut[{i}]: {df_diag.filter(DESI_IS_NORTH & expr).height}")
    for i, expr in enumerate(DESI_CUTS_S):
        print(f"  SOUTH cut[{i}]: {df_diag.filter(DESI_IS_SOUTH & expr).height}")
    print(f"  LRG combined:  {df_diag.filter(DESI_LRG_CUT).height}")
    print("--- END DIAGNOSTICS ---\n")

    # ── full collect ──────────────────────────────────────────────────────────
    print("Applying LRG cuts and collecting...")
    df_lrg = lf_base.filter(DESI_LRG_CUT).collect()
    print(f"  selected: {len(df_lrg):,} LRG galaxies")

    if len(df_lrg) == 0:
        print("ERROR: 0 galaxies — check cut diagnostics above.")
        import sys; sys.exit(1)

    # ── mag vs redshift ───────────────────────────────────────────────────────
    z = df_lrg[PARAMS["redshift_col"]].to_numpy()
    for mag_col in PARAMS["mag_cols"]:
        print(f"  plotting {mag_col}...")
        mag = df_lrg[mag_col].to_numpy()
        print(f"    range: {np.nanmin(mag):.2f} – {np.nanmax(mag):.2f}  nulls: {np.isnan(mag).sum()}")
        fig = plot_mag_vs_redshift(z, mag, mag_col, PARAMS["redshift_col"])
        run.save_plot(fig, f"{mag_col}_vs_{PARAMS['redshift_col']}_lrg.png")
        plt.close(fig)

    # ── colour planes ─────────────────────────────────────────────────────────
    def colours(df, col_a, col_b):
        arr = (df[col_a] - df[col_b]).to_numpy()
        print(f"    {col_a}-{col_b}: range {np.nanmin(arr):.2f} – {np.nanmax(arr):.2f}  nulls: {np.isnan(arr).sum()}")
        return arr

    print("Colour plane: z-W1 vs r-z")
    fig = plot_colour_plane(
        colours(df_lrg, "lsst_r", "lsst_z"),
        colours(df_lrg, "lsst_z", "wise_w1_mag"),
        "r - z", "z - W1", cut_fn=_cut_zw1_rz,
        title="z-W1 vs r-z  (LRG selected)",
    )
    run.save_plot(fig, "colour_zw1_vs_rz.png")
    plt.close(fig)

    print("Colour plane: g-W1 vs r-W1")
    fig = plot_colour_plane(
        colours(df_lrg, "lsst_r", "wise_w1_mag"),
        colours(df_lrg, "lsst_g", "wise_w1_mag"),
        "r - W1", "g - W1",
        title="g-W1 vs r-W1  (LRG selected)",
    )
    run.save_plot(fig, "colour_gw1_vs_rw1.png")
    plt.close(fig)

    print("Colour plane: r-W1 vs W1")
    w1  = df_lrg["wise_w1_mag"].to_numpy()
    rw1 = colours(df_lrg, "lsst_r", "wise_w1_mag")
    valid = np.isfinite(w1) & np.isfinite(rw1)
    print(f"    valid points: {valid.sum()}")
    fig, ax = plt.subplots(figsize=(7, 6))
    rng = np.random.default_rng(42)
    idx = rng.choice(valid.sum(), min(50_000, valid.sum()), replace=False)
    ax.scatter(w1[valid][idx], rw1[valid][idx], s=0.3, alpha=0.3, c="steelblue")
    w1_range = np.linspace(*np.nanpercentile(w1[valid], [1, 99]), 500)
    ax.plot(w1_range, _cut_rw1_w1_north(w1_range), color="red",    lw=1.5, label="North cut")
    ax.plot(w1_range, _cut_rw1_w1_south(w1_range), color="orange", lw=1.5, label="South cut")
    ax.set_xlabel("W1")
    ax.set_ylabel("r - W1")
    ax.set_title("r-W1 vs W1  (LRG selected)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    run.save_plot(fig, "colour_rw1_vs_w1.png")
    plt.close(fig)

    run.save_result(
        df_lrg.select(PARAMS["mag_cols"] + [PARAMS["redshift_col"]]).describe(),
        "summary_stats_lrg.csv"
    )
    print("Done!")
