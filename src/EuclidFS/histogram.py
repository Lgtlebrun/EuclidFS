# histogram.py
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import polars as pl
from .data import iter_buckets

@dataclass
class HistSpec:
    x:       pl.Expr
    y:       pl.Expr
    x_bins:  np.ndarray
    y_bins:  np.ndarray
    x_label: str
    y_label: str

@dataclass
class Hist2D:
    counts:  np.ndarray
    x_bins:  np.ndarray
    y_bins:  np.ndarray
    x_label: str
    y_label: str
    n_total: int

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        mesh = ax.pcolormesh(
            self.x_bins, self.y_bins, self.counts.T,
            cmap="inferno", **kwargs
        )
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(f"{self.y_label} vs {self.x_label}  (N={self.n_total:,})")
        return ax, mesh

def compute_hist2d(
    spec:          HistSpec,
    prepare:       Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    bucket_ids:    list[int] | None = None,
    target_ram_gb: float | None = None,
) -> Hist2D:
    counts  = np.zeros((len(spec.x_bins) - 1, len(spec.y_bins) - 1), dtype=np.int64)
    n_total = 0

    # wrap prepare to also materialise x/y as named columns
    def _prepare(lf: pl.LazyFrame) -> pl.LazyFrame:
        if prepare is not None:
            lf = prepare(lf)
        return lf.with_columns([
            spec.x.alias("__x__"),
            spec.y.alias("__y__"),
        ]).select(["__x__", "__y__"])

    for chunk in iter_buckets(bucket_ids, prepare=_prepare, target_ram_gb=target_ram_gb):
        x    = chunk["__x__"].to_numpy()
        y    = chunk["__y__"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        h, _, _ = np.histogram2d(x[mask], y[mask], bins=[spec.x_bins, spec.y_bins])
        counts  += h.astype(np.int64)
        n_total += int(mask.sum())

    return Hist2D(
        counts  = counts,
        x_bins  = spec.x_bins,
        y_bins  = spec.y_bins,
        x_label = spec.x_label,
        y_label = spec.y_label,
        n_total = n_total,
    )