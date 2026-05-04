import polars as pl
import psutil
from pathlib import Path
from typing import Callable, Iterator

BUCKETS_PATH = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8_with_w")
BUCKETS      = range(23070, 23078)

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