from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import polars as pl
from tqdm import tqdm
from .data import iter_files, _iter_lazyframe, DATA_FILES
from typing import Callable
from pathlib import Path

class _BaseHist:
    """
    Shared chunked-accumulation logic for all histogram types.
    Subclasses implement `_accumulate(chunk_arrays)` and `_result()`.
    """

    def __init__(
        self,
        filters:       list[pl.Expr] | None = None,
        prepare_fn:    Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
        files:    list[Path] | None = None,
        lf:            pl.LazyFrame | None = None,
        target_ram_gb: float | None = None,
    ):
        self.prepare_fn    = prepare_fn
        self.filters       = filters    or []
        self.files    = files or DATA_FILES
        self.lf            = lf
        self.target_ram_gb = target_ram_gb

    def _prepare(self, exprs: list[pl.Expr]) -> Callable:
        """Returns a prepare function that injects filters, and histogram exprs."""
        def prepare(lf: pl.LazyFrame) -> pl.LazyFrame:
            for f in self.filters:
                lf = lf.filter(f)
            if self.prepare_fn is not None:
                lf = self.prepare_fn(lf)
            #For example : [pl.col("sdss_r_mag") - pl.col("wise_w1_mag")).alias("__col0__"),
            #    pl.col("true_redshift_gal").alias("__col1__")]
            col_names = [f"__col{i}__" for i in range(len(exprs))]
            return (
                lf.with_columns([e.alias(name) for e, name in zip(exprs, col_names)])
                  .select(col_names)
            )
        return prepare

    def _run(self, exprs: list[pl.Expr], desc: str) -> None:
        prepare   = self._prepare(exprs)
        col_names = [f"__col{i}__" for i in range(len(exprs))]

        if self.lf is not None:
            # use provided lazy frame directly — no file scan
            chunks = _iter_lazyframe(self.lf, prepare=prepare,
                                     target_ram_gb=self.target_ram_gb)
            total  = 1  # tqdm unknown total
        else:
            chunks = iter_files(self.files, prepare=prepare,
                                target_ram_gb=self.target_ram_gb)
            total  = len(self.files)

        with tqdm(total=total, desc=desc, unit="chunk") as pbar:
            for chunk in chunks:
                arrays = [chunk[c].to_numpy() for c in col_names]
                self._accumulate(arrays)
                pbar.update(1)
                pbar.set_postfix(chunk_rows=f"{len(arrays[0]):,}")

    def _accumulate(self, arrays: list[np.ndarray]) -> None:
        raise NotImplementedError

    def _result(self):
        raise NotImplementedError

@dataclass
class Hist1D(_BaseHist):
    """
    1D histogram accumulated in RAM-safe chunks over the full catalog.

    Usage
    -----
        h = Hist1D(
                expr    = pl.col("true_redshift_gal"),
                bins    = np.linspace(0, 1.3, 201),
                label   = "redshift",
                filters = [pl.col("true_redshift_gal") < 1.3],
            ).compute()
        h.plot()
    """
    expr:          pl.Expr
    bins:          np.ndarray
    label:         str
    filters:       list[pl.Expr]  = field(default_factory=list)
    prepare_fn:    Callable | None       = None
    files:         list[Path]      | None = None
    lf:            pl.LazyFrame    | None = None 
    target_ram_gb: float          | None = None

    # results — populated by compute()
    counts:  np.ndarray | None = field(default=None, init=False)
    n_total: int               = field(default=0,    init=False)

    def __post_init__(self):
        _BaseHist.__init__(self, self.filters, self.prepare_fn,
                           self.files,  self.lf, self.target_ram_gb)
        self._counts = np.zeros(len(self.bins) - 1, dtype=np.int64)

    def _accumulate(self, arrays):
        x    = arrays[0]
        mask = np.isfinite(x)
        h, _ = np.histogram(x[mask], bins=self.bins)
        self._counts += h.astype(np.int64)
        self.n_total += int(mask.sum())

    def compute(self) -> Hist1D:
        self._run([self.expr], desc=self.label)
        self.counts = self._counts
        return self

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.stairs(self.counts, self.bins, **kwargs)
        ax.set_xlabel(self.label)
        ax.set_ylabel("count")
        ax.set_title(f"{self.label}  (N={self.n_total:,})")
        return fig, ax


@dataclass
class Hist2D(_BaseHist):
    x:             pl.Expr
    y:             pl.Expr
    x_bins:        np.ndarray
    y_bins:        np.ndarray
    x_label:       str
    y_label:       str
    filters:       list[pl.Expr]   = field(default_factory=list)
    prepare_fn:    Callable | None = None
    files:         list[Path]      | None = None
    lf:            pl.LazyFrame    | None = None  
    target_ram_gb: float           | None = None

    counts:  np.ndarray | None = field(default=None, init=False)
    n_total: int               = field(default=0,    init=False)

    def __post_init__(self):
        _BaseHist.__init__(self, self.filters, self.prepare_fn,
                           self.files, self.lf, self.target_ram_gb)
        self._counts = np.zeros((len(self.x_bins)-1, len(self.y_bins)-1), dtype=np.int64)
    # rest unchanged

    def _accumulate(self, arrays):
        x, y = arrays
        mask = np.isfinite(x) & np.isfinite(y)
        h, _, _ = np.histogram2d(x[mask], y[mask], bins=[self.x_bins, self.y_bins])
        self._counts += h.astype(np.int64)
        self.n_total += int(mask.sum())

    def compute(self) -> Hist2D:
        self._run([self.x, self.y], desc=f"{self.x_label} vs {self.y_label}")
        self.counts = self._counts
        return self

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        mesh = ax.pcolormesh(self.x_bins, self.y_bins, self.counts.T,
                             cmap="inferno", **kwargs)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(f"{self.y_label} vs {self.x_label}  (N={self.n_total:,})")
        return ax, mesh