import polars as pl
import psutil, os
from pathlib import Path
from typing import Callable, Iterator
# from dotenv import load_dotenv
# from .config import _find_dotenv

#Locate the Root relative to this file
# __file__ is the path to data.py
# PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
# print(f"DEBUG : PACKAGE_ROOT IS : {PACKAGE_ROOT}")
# DOTENV_PATH = _find_dotenv()
# print(f".env path considered : {DOTENV_PATH}")

# #Load it
# if DOTENV_PATH.exists():
#     load_dotenv(DOTENV_PATH)
# else:
#     # This helps you debug in the Notebook if the file isn't found
#     print(f"Warning: .env not found at {DOTENV_PATH}")

# Fetch with fallbacks (the second argument is a default if .env is missing)
# _path_str = os.getenv("BUCKETS_PATH", "./data")
# BUCKETS_PATH = Path(_path_str)
DATA_PATH = Path(os.environ["DATA_PATH"])
# _start = int(os.getenv("BUCKET_START", 23070))
# _end = int(os.getenv("BUCKET_END", 23078))
# BUCKETS = range(_start, _end) if _start != _end else [_start]    # Allow single bucket
# DATA_FILES = []

# for bucket in BUCKETS :
#     DATA_FILES.extend(list(BUCKETS_PATH.glob(f"{bucket}_part_*.parquet")))

DATA_FILES = sorted(list(DATA_PATH.glob("*.parquet")))

FLUX_COLUMNS = ["euclid_nisp_h", 
                'euclid_nisp_h_abs', 
                'lsst_g',
                'lsst_i',
                'lsst_r',
                'lsst_u',
                'lsst_y',
                'lsst_z',
                'sdss_g',
                'sdss_i',
                'sdss_r',
                'sdss_u',
                'sdss_z',
                'wise_w1',
                'wise_w2',
                ]

FLUX2MAG_MAP = {flux : f"{flux}_mag" for flux in FLUX_COLUMNS}
MAG2FLUX_MAP = {mag : flux for flux, mag in FLUX2MAG_MAP.items()}
MAG_COLUMNS = list(FLUX2MAG_MAP.values())

RAM_SAFETY_FACTOR = float(os.getenv("RAM_SAFETY_FACTOR", 0.6))

def available_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9

def estimate_chunk_size(sample: pl.DataFrame, target_gb: float) -> int:
    bytes_per_row = sample.estimated_size() / len(sample)
    return max(1000, int(target_gb * 1e9 / bytes_per_row))

def convert_flux_to_mag(ldf:pl.LazyFrame, column : str) -> pl.LazyFrame:
    if not column in ldf.collect_schema().names():
        raise KeyError("WARNING : column not in ldf, cannnot convert to mag")
    
    return ldf.with_columns([
            pl.when(pl.col(column) > 0)
            .then(-2.5 * pl.col(column).log(base=10) - 48.6)
            .otherwise(None)
            .alias(f"{column}_mag"),
        ])

def load_lazy(
    files:      list[Path] = DATA_FILES,
    filters:    list[pl.Expr] | None = None,
    select:     list[str] | None = None,
    n_rows : int | None = None,
    prepare : Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
) -> pl.LazyFrame:

    lf = pl.scan_parquet(files)

    if filters is not None:
        lf = lf.filter(pl.all_horizontal(filters))

    if select is not None:
        lf = lf.select(select)

    if prepare is not None:
        lf = prepare(lf)

    print("DEBUG : load_lazy : prepared !")

    if n_rows is not None:
        # Polars will stop reading files as soon as it has n_rows that pass 'prepare'
        lf = lf.head(n_rows)
        print("DEBUG: lf headed")

    return lf

def iter_chunks(
    files: list[Path] | None = None,
    lf: pl.LazyFrame | None = None,
    prepare: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    target_ram_gb: float | None = None,
) -> Iterator[pl.DataFrame]:
    """
    Consolidated chunker: Scans all files at once or uses a provided LazyFrame
    to yield chunks that actually fill the target RAM.
    """
    # 1. Setup the source
    if lf is None:
        files = files or DATA_FILES
        # SCAN ALL FILES AT ONCE - This is the key change
        lf = pl.scan_parquet(files)

    # 2. Apply transformations
    if prepare is not None:
        lf = prepare(lf)

    # 3. Calculate actual chunk size based on target RAM
    target_gb = target_ram_gb or (available_ram_gb() * RAM_SAFETY_FACTOR)
    
    # We collect a small sample to see how 'heavy' each row is after preparation
    sample = lf.limit(10_000).collect()
    if len(sample) == 0:
        return

    chunk_size = estimate_chunk_size(sample, target_gb)
    print(f"DEBUG: Optimized Chunk Size = {chunk_size:,} rows (~{target_gb:.1f} GB)")

    # 4. Stream the giant LazyFrame in massive chunks
    offset = 0
    while True:
        # Polars handles the parallel disk I/O across multiple files here
        chunk = lf.slice(offset, chunk_size).collect()
        
        if len(chunk) == 0:
            break
            
        yield chunk
        
        offset += chunk_size
        del chunk # Explicitly clear for the next big allocation
        

def _iter_lazyframe(
    lf:            pl.LazyFrame,
    prepare:       Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    target_ram_gb: float | None = None,
) -> Iterator[pl.DataFrame]:
    target_gb = target_ram_gb or (available_ram_gb() * RAM_SAFETY_FACTOR)

    if prepare is not None:
        lf = prepare(lf)

    sample = lf.limit(1000).collect()
    if len(sample) == 0:
        return

    chunk_size = estimate_chunk_size(sample, target_gb)
    offset     = 0
    while True:
        chunk = lf.slice(offset, chunk_size).collect()
        if len(chunk) == 0:
            break
        yield chunk
        offset += chunk_size
        del chunk

def random_sample_lazy(
    n:       int = 50_000,
    files:   list[Path] | None = None,
    prepare: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
) -> pl.LazyFrame:
    """
    Returns a lazy frame of ~n rows using random_index.
    """
    files = files or DATA_FILES
    total = pl.scan_parquet(files).select(pl.len()).collect().item()
    frac  = min(n / total, 1.0)
    print(f"  total rows: {total:,}  →  fraction: {frac:.6f}")

    lf = pl.scan_parquet(files)

    if prepare is not None:
        lf = prepare(lf)

    return lf.filter(pl.col("random_index") < frac)


def random_sample(
    n:             int = 50_000,
    files:         list[Path] | None = None,
    prepare:       Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    target_ram_gb: float | None = None,
) -> pl.DataFrame:
    files     = files or DATA_FILES
    total     = pl.scan_parquet(files).select(pl.len()).collect().item()
    frac      = min(n / total, 1.0)
    print(f"  total rows: {total:,}  →  fraction: {frac:.6f}")

    def _prepare(lf: pl.LazyFrame) -> pl.LazyFrame:
        lf = lf.filter(pl.col("random_index") < frac)
        if prepare is not None:
            lf = prepare(lf)
        return lf

    return pl.concat(list(iter_files(files=files, prepare=_prepare,
                                     target_ram_gb=target_ram_gb)))