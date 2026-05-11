import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from EuclidFS.data import load_lazy
from EuclidFS.colour_cuts import apply_lrg_cuts, DESI_LRG_COLUMNS
from EuclidFS.config import RunDir

REDSHIFT_COL = "true_redshift_gal"
ZMAX         = 1.3
COLUMNS = DESI_LRG_COLUMNS | {REDSHIFT_COL, "random_index"}
N_SCATTER    = 50_000

# ── redshift colorbar setup ───────────────────────────────────────────────────
NUM_BINS     = 12
CMAP         = plt.get_cmap("Paired")
BOUNDS       = np.linspace(0.0, ZMAX, NUM_BINS + 1)
NORM         = BoundaryNorm(BOUNDS, CMAP.N)


# ── colour helpers ────────────────────────────────────────────────────────────

def colours(df, col_a, col_b):
    arr = (df[col_a] - df[col_b]).to_numpy()
    print(f"    {col_a}-{col_b}: range {np.nanmin(arr):.2f} – {np.nanmax(arr):.2f}  nulls: {np.isnan(arr).sum()}")
    return arr

def scatter_with_redshift(ax, x, y, z, title, xlabel, ylabel):
    valid    = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    sort_idx = np.argsort(z[valid])
    sc = ax.scatter(x[valid][sort_idx], y[valid][sort_idx],
                    s=0.8, alpha=0.7, c=z[valid][sort_idx],
                    cmap=CMAP, norm=NORM, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return sc, valid.sum()


# ── cut boundary functions ────────────────────────────────────────────────────

def boundary_rw1_w1(w1, region="north"):
    if region == "north":
        calc_val, cond1, default = 1.83 * (w1 - 17.13), w1 - 16.31, 3.4
    else:
        calc_val, cond1, default = 1.8  * (w1 - 17.14), w1 - 16.33, 3.3
    return w1, np.minimum(np.maximum(calc_val, cond1), default)

def boundary_rw1_gr(rw1, region="north"):
    rw1  = np.sort(rw1[rw1 <= 1.8])
    rw1  = np.append(rw1, 1.8)
    y    = np.where(region == "north", -rw1 + 2.97, -rw1 + 2.9)
    y[-1] = 0
    return rw1, y

def boundary_rz_zw1(rz):
    return rz, 0.8 * rz - 0.6


# ── individual plot functions ─────────────────────────────────────────────────

def plot_rw1_vs_w1(random, redshift_col):
    w1  = random["wise_w1_mag"].to_numpy()
    rw1 = colours(random, "sdss_r_mag", "wise_w1_mag")
    z   = random[redshift_col].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    sc, n = scatter_with_redshift(ax, w1, rw1, z, "r-W1 vs W1  (LRG selected)", "W1", "r - W1")
    cbar  = fig.colorbar(sc, ax=ax, boundaries=BOUNDS, ticks=BOUNDS)
    cbar.set_label("Redshift")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    w1_range     = np.linspace(*np.nanpercentile(w1[np.isfinite(w1)], [1, 99]), 500)
    w1_n, bn     = boundary_rw1_w1(w1_range, region="north")
    w1_s, bs     = boundary_rw1_w1(w1_range, region="south")
    ax.plot(w1_n, bn, color="red",    lw=1.5, label="North cut")
    ax.plot(w1_s, bs, color="orange", lw=1.5, label="South cut")
    ax.legend(fontsize=8)
    plt.tight_layout()
    print(f"    r-W1 vs W1: {n:,} valid points")
    return fig

def plot_gr_vs_rw1(random, redshift_col):
    gr  = colours(random, "sdss_g_mag", "sdss_r_mag")
    rw1 = colours(random, "sdss_r_mag", "wise_w1_mag")
    z   = random[redshift_col].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    sc, n   = scatter_with_redshift(ax, rw1, gr, z, "g-r vs r-W1  (LRG selected)", "r - W1", "g - r")
    cbar    = fig.colorbar(sc, ax=ax, boundaries=BOUNDS, ticks=BOUNDS)
    cbar.set_label("Redshift")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    rw1_range    = np.linspace(0.5, 2.5, 500)
    x_n, y_n     = boundary_rw1_gr(rw1_range, region="north")
    x_s, y_s     = boundary_rw1_gr(rw1_range, region="south")
    ax.plot(x_n, y_n, color="red",    lw=2, label="North cut")
    ax.plot(x_s, y_s, color="orange", lw=2, label="South cut")
    ax.vlines(1.8, ymin=ax.get_ylim()[0], ymax=2.97 - 1.8, color="red",    lw=2, linestyle="--")
    ax.vlines(1.8, ymin=ax.get_ylim()[0], ymax=2.9  - 1.8, color="orange", lw=2, linestyle="--")
    ax.legend(fontsize=8)
    plt.tight_layout()
    print(f"    g-r vs r-W1: {n:,} valid points")
    return fig

def plot_zw1_vs_rz(random, redshift_col):
    zw1 = colours(random, "sdss_z_mag", "wise_w1_mag")
    rz  = colours(random, "sdss_r_mag", "sdss_z_mag")
    z   = random[redshift_col].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    sc, n   = scatter_with_redshift(ax, rz, zw1, z, "z-W1 vs r-z  (LRG selected)", "r - z", "z - W1")
    cbar    = fig.colorbar(sc, ax=ax, boundaries=BOUNDS, ticks=BOUNDS)
    cbar.set_label("Redshift")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    rz_range = np.linspace(0.5, 2.5, 500)
    x, y     = boundary_rz_zw1(rz_range)
    ax.plot(x, y, color="red", lw=2, label="cut boundary")
    ax.legend(fontsize=8)
    plt.tight_layout()
    print(f"    z-W1 vs r-z: {n:,} valid points")
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run = RunDir("lrg_colour_cuts")

    print(f"Loading and applying LRG cuts with redshift z < {ZMAX}...")
    df_lrg = (
        load_lazy(n_rows = N_SCATTER, prepare=apply_lrg_cuts, filters=[pl.col(REDSHIFT_COL) < ZMAX], select=list(COLUMNS))
        .collect(streaming=True)
    )
    print(f"  {len(df_lrg):,} LRG galaxies selected")

    if len(df_lrg) == 0:
        print("ERROR: no galaxies selected — check cuts.")
        sys.exit(1)

    random = df_lrg.sort("random_index").head(N_SCATTER)

    print("Plotting r-W1 vs W1...")
    run.save_plot(plot_rw1_vs_w1(random, REDSHIFT_COL),   "rw1_vs_w1.png")
    plt.close()

    print("Plotting g-r vs r-W1...")
    run.save_plot(plot_gr_vs_rw1(random, REDSHIFT_COL),   "gr_vs_rw1.png")
    plt.close()

    print("Plotting z-W1 vs r-z...")
    run.save_plot(plot_zw1_vs_rz(random, REDSHIFT_COL),   "zw1_vs_rz.png")
    plt.close()

    run.save_result(
        df_lrg.select(["wise_w1_mag", "sdss_r_mag", "sdss_g_mag",
                        "sdss_z_mag", REDSHIFT_COL]).describe(),
        "summary_stats.csv",
    )
    print("Done!")