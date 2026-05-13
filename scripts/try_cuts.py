import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from EuclidFS.config import RunDir
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from EuclidFS.colour_cuts import apply_lrg_cuts
from EuclidFS.data import load_lazy
import argparse

Z_CUTS   = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
MAG_CUTS = list(range(14, 31))          # 14 … 30 inclusive

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows",       type=int,   default=int(1e6))
    parser.add_argument("--redshift-max", type=float, default=3.0)
    parser.add_argument("--mag-max",      type=float, default=30)
    args = parser.parse_args()

    run          = RunDir("try_cuts")
    redshift_col = "true_redshift_gal"
    mag_col      = "wise_w1_mag"
    select_cols  = [
        "ra_gal", "dec_gal", "abs_mag_r01", "euclid_nisp_h_mag",
        "sdss_r_mag", "sdss_g_mag", "sdss_z_mag", "sdss_i_mag",
        "wise_w1_mag", "wise_w2_mag", redshift_col, "random_index",
    ]

    # ── collect once ──────────────────────────────────────────────────────
    print("Collecting base sample...")
    df_base = load_lazy(
        n_rows=args.n_rows,
        select=select_cols,
        filters=[pl.col(redshift_col) < 3.0],
    ).collect(streaming=True)
    n_base = len(df_base)
    print(f"Base (z < 3): {n_base:,}")

    # ── 1-D: fraction surviving z cut ────────────────────────────────────
    z_fracs = np.array([
        df_base.filter(pl.col(redshift_col) < z).height / n_base
        for z in Z_CUTS
    ])

    # ── 1-D: fraction surviving mag cut ──────────────────────────────────
    mag_fracs = np.array([
        df_base.filter(pl.col(mag_col) < m).height / n_base
        for m in MAG_CUTS
    ])

    # ── 2-D: fraction surviving z AND mag cut (heatmap) ──────────────────
    z_grid   = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    mag_grid = np.array(MAG_CUTS, dtype=float)
    heat     = np.zeros((len(z_grid), len(mag_grid)))
    for i, z in enumerate(z_grid):
        for j, m in enumerate(mag_grid):
            heat[i, j] = (
                df_base.filter(
                    (pl.col(redshift_col) < z) & (pl.col(mag_col) < m)
                ).height / n_base
            )

    # ── LRG cut ───────────────────────────────────────────────────────────
       # ── apply LRG cuts once ───────────────────────────────────────────────
    df_lrg = apply_lrg_cuts(df_base.lazy()).collect()
    n_lrg  = len(df_lrg)
    f_lrg  = n_lrg / n_base
    print(f"DESI LRG cut: {n_lrg:,}  ({100*f_lrg:.1f}%)")

    # ── 1-D LRG: fraction surviving z cut ────────────────────────────────
    z_fracs_lrg = np.array([
        df_lrg.filter(pl.col(redshift_col) < z).height / n_base
        for z in Z_CUTS
    ])

    # ── 1-D LRG: fraction surviving mag cut ──────────────────────────────
    mag_fracs_lrg = np.array([
        df_lrg.filter(pl.col(mag_col) < m).height / n_base
        for m in MAG_CUTS
    ])

    # ── 2-D LRG heatmap ───────────────────────────────────────────────────
    heat_lrg = np.zeros((len(z_grid), len(mag_grid)))
    for i, z in enumerate(z_grid):
        for j, m in enumerate(mag_grid):
            heat_lrg[i, j] = (
                df_lrg.filter(
                    (pl.col(redshift_col) < z) & (pl.col(mag_col) < m)
                ).height / n_base
            )

    # ══ PLOTS ═════════════════════════════════════════════════════════════
    def plot_1d_z(fracs, n, label, fname):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(range(len(Z_CUTS)), fracs * 100, color="#3a86ff", alpha=0.85)
            ax.set_xticks(range(len(Z_CUTS)))
            ax.set_xticklabels([f"z<{z}" for z in Z_CUTS])
            ax.set_ylabel("Fraction (%)")
            ax.set_title(f"Redshift cuts  —  {label}  N = {n:,}")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            fig.tight_layout()
            run.save_plot(fig, fname)

    def plot_1d_mag(fracs, n, label, fname):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(MAG_CUTS, fracs * 100, marker="o", color="#3a86ff", lw=2)
        ax.set_xlabel(mag_col)
        ax.set_ylabel("Fraction (%)")
        ax.set_title(f"{mag_col} cuts  —  {label}  N = {n:,}")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        run.save_plot(fig, fname)

    def plot_2d_heat(heat, n, label, fname):
        fig, ax = plt.subplots(figsize=(10, 5))
        vmin = np.min(heat)
        vmax = np.max(heat)
        im = ax.imshow(
            heat * 100, aspect="auto", origin="lower", cmap="plasma",
            vmin=vmin, vmax=vmax,
            extent=[mag_grid[0]-0.5, mag_grid[-1]+0.5,
                    z_grid[0]-0.25,  z_grid[-1]+0.25],
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Fraction (%)")
        ax.set_xlabel(mag_col)
        ax.set_ylabel("redshift max")
        ax.set_yticks(z_grid)
        ax.set_title(f"Fraction surviving z & {mag_col} cuts  —  {label}  N = {n:,}")
        for i, z in enumerate(z_grid):
            for j, m in enumerate(mag_grid):
                ax.text(m, z, f"{heat[i,j]*100:.0f}",
                        ha="center", va="center",
                        fontsize=6, color="white" if heat[i,j] < 0.6 else "black")
        fig.tight_layout()
        run.save_plot(fig, fname)

    # ── base plots ────────────────────────────────────────────────────────
    plot_1d_z  (z_fracs,       n_base, "base",     "1d_z_cuts")
    plot_1d_mag(mag_fracs,     n_base, "base",     "1d_mag_cuts")
    plot_2d_heat(heat,         n_base, "base",     "2d_heatmap_z_vs_mag")

    # ── LRG plots ─────────────────────────────────────────────────────────
    plot_1d_z  (z_fracs_lrg,   n_lrg,  "LRG cuts", "1d_z_cuts_lrg")
    plot_1d_mag(mag_fracs_lrg, n_lrg,  "LRG cuts", "1d_mag_cuts_lrg")
    plot_2d_heat(heat_lrg,     n_lrg,  "LRG cuts", "2d_heatmap_z_vs_mag_lrg")

    # ── LRG summary bar ───────────────────────────────────────────────────
    fig4, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["All", "DESI LRG"], [100, f_lrg * 100], color=["#8ecae6", "#e63946"])
    ax.set_ylabel("Fraction of base sample (%)")
    ax.set_title(f"DESI LRG colour cut  —  base N = {n_base:,}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    fig4.tight_layout()
    run.save_plot(fig4, "lrg_cut_summary")

    print("Done!")