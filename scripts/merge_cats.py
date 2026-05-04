import polars as pl
from pathlib import Path
import gc

BUCKETS_PATH = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8")
EXTRA_PATH   = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8/25697.parquet")
BUCKETS      = range(23070, 23078)
OUT_PATH     = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8_with_w")

OUT_PATH.mkdir(exist_ok=True)

extra = pl.scan_parquet(EXTRA_PATH)

for i in BUCKETS:
    print(f"Merging bucket {i}...")
    bucket = pl.scan_parquet(BUCKETS_PATH / f"{i}.parquet")
    merged = (
        bucket
        .join(extra, on=["galaxy_id", "halo_id"], how="left")
        .collect()
    )
    merged.write_parquet(OUT_PATH / f"{i}.parquet")
    del merged
    gc.collect()
    print(f"  {len(merged):,} rows, {merged.width} cols")

print("Done!")
