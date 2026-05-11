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
BUCKETS_PATH = Path(os.environ["BUCKETS_PATH"])

_start = int(os.getenv("BUCKET_START", 23070))
_end = int(os.getenv("BUCKET_END", 23078))
BUCKETS = range(_start, _end) if _start != _end else [_start]    # Allow single bucket

FLUX_COLUMNS = ["euclid_nisp_h", 
                'euclid_nisp_h_abs', 
                'lsst_g',
                'lsst_i',
                'lsst_r',
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
    bucket_ids: list[int] | None = None,
    filters:    list[pl.Expr] | None = None,
    select:     list[str] | None = None,
    n_rows : int | None = None,
    prepare : Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
) -> pl.LazyFrame:
    paths = [
        BUCKETS_PATH / f"{i}.parquet"
        for i in (bucket_ids or BUCKETS)
    ]

    # First drop useless rows
    if n_rows is not None :

        total = pl.scan_parquet(paths).select(pl.len()).collect().item()
        frac = min(n_rows / total, 1.0)
        print(f"  total rows: {total:,}, selecting {n_rows}  →  fraction: {frac:.6f}")
        lf = pl.scan_parquet(paths).filter(pl.col("random_index") < frac)
    else:
        lf = pl.scan_parquet(paths)

    if filters is not None:
        for f in filters:
            lf = lf.filter(f)

    if select is not None:
        if isinstance(select, str) : select = [select]   # allow 1 col

        # Add magnitude
        for col in select :
            if col in MAG_COLUMNS:
                col_flux = MAG2FLUX_MAP[col]
                lf = convert_flux_to_mag(lf, col_flux)
                
        lf = lf.select(select)
    else :
        # Add magnitude
        for col in FLUX_COLUMNS:
            lf = convert_flux_to_mag(lf, col)

    if prepare is not None:
        lf = prepare(lf)

    return lf

def iter_buckets(
    bucket_ids:    list[int] | None = None,
    prepare:       Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    target_ram_gb: float | None = None,
) -> Iterator[pl.DataFrame]:
    """
    Yields chunks of data across all buckets, RAM-safe.
    `prepare` is a function that takes a LazyFrame and returns a LazyFrame —
    use it to add columns, apply cuts, select, etc.
    """
    # If not indicated, uses available ram x safety factor
    target_gb = target_ram_gb or (available_ram_gb() * RAM_SAFETY_FACTOR)

    for bucket_id in (bucket_ids or BUCKETS):
        print(f"  bucket {bucket_id}  (RAM free: {available_ram_gb():.1f} GB)")

        lf = pl.scan_parquet(BUCKETS_PATH / f"{bucket_id}.parquet")

        # same mag conversion as load_lazy — always applied to all flux cols
        for col in FLUX_COLUMNS:
            lf = convert_flux_to_mag(lf, col)

        if prepare is not None:
            lf = prepare(lf)

        # probe
        sample = lf.limit(1000).collect()
        if len(sample) == 0:
            print(f"    skipping — empty after prepare()")
            continue

        chunk_size = estimate_chunk_size(sample, target_gb)
        print(f"    chunk size: {chunk_size:,}")

        offset = 0
        while True:
            chunk = lf.slice(offset, chunk_size).collect()
            if len(chunk) == 0:
                break
            yield chunk
            offset += chunk_size
            del chunk

def random_sample(
    n:             int = 50_000,
    bucket_ids:    list[int] | None = None,
    prepare:       Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    target_ram_gb: float | None = None,
) -> pl.DataFrame:
    """
    Draw a uniform sample of ~n rows using the pre-computed random_index column.
    Requires random_index in [0, 1) to be present in the catalog.
    """
    # estimate the fraction needed to get ~n rows
    # we need total count first — cheap since it's just a count aggregation
    bucket_paths = [BUCKETS_PATH / f"{i}.parquet" for i in (bucket_ids or BUCKETS)]
    total = pl.scan_parquet(bucket_paths).select(pl.len()).collect().item()

    frac = min(n / total, 1.0)
    print(f"  total rows: {total:,}  →  sampling fraction: {frac:.6f}")

    def _prepare(lf: pl.LazyFrame) -> pl.LazyFrame:
        # First apply optional preparation
        if prepare is not None:
            lf = prepare(lf)
        # Then filter on random index
        return lf.filter(pl.col("random_index") < frac)

    chunks = list(iter_buckets(
        bucket_ids    = bucket_ids,
        prepare       = _prepare,
        target_ram_gb = target_ram_gb,
    ))
    return pl.concat(chunks)
