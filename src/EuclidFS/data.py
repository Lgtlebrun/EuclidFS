import polars as pl
import psutil, os
from pathlib import Path
from typing import Callable, Iterator
from dotenv import load_dotenv

#Locate the Root relative to this file
# __file__ is the path to data.py
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = PACKAGE_ROOT / ".env"

#Load it
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)
else:
    # This helps you debug in the Notebook if the file isn't found
    print(f"Warning: .env not found at {DOTENV_PATH}")

# Fetch with fallbacks (the second argument is a default if .env is missing)
_path_str = os.getenv("BUCKETS_PATH", "./data")
BUCKETS_PATH = Path(_path_str)

_start = int(os.getenv("BUCKET_START", 23070))
_end = int(os.getenv("BUCKET_END", 23078))
BUCKETS = range(_start, _end) if _start != _end else [_start]

def available_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9

def estimate_chunk_size(sample: pl.DataFrame, target_gb: float) -> int:
    bytes_per_row = sample.estimated_size() / len(sample)
    return max(1000, int(target_gb * 1e9 / bytes_per_row))

def load_lazy(
    bucket_ids: list[int] | None = None,
    filters:    list[pl.Expr] | None = None,
    select:     list[str] | None = None,
) -> pl.LazyFrame:
    paths = [
        BUCKETS_PATH / f"{i}.parquet"
        for i in (bucket_ids or BUCKETS)
    ]
    lf = pl.scan_parquet(paths)
    if filters:
        for f in filters:
            lf = lf.filter(f)
    if select:
        lf = lf.select(select)
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
    target_gb = target_ram_gb or (available_ram_gb() * 0.6)

    for bucket_id in (bucket_ids or BUCKETS):
        print(f"  bucket {bucket_id}  (RAM free: {available_ram_gb():.1f} GB)")

        lf = pl.scan_parquet(BUCKETS_PATH / f"{bucket_id}.parquet")

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