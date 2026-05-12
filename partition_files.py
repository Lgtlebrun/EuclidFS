import polars as pl
from pathlib import Path
from tqdm import tqdm
import glob
import os

def repartition_files(
    input_glob:    str,
    output_dir:    Path,
    n_partitions:  int = 100,   # split [0,1) into n equal slices
):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(input_glob))
    if not files:
        print(f"No files found matching: {input_glob}")
        return
    print(f"Found {len(files)} input files → {n_partitions} output partitions")

    edges = [i / n_partitions for i in range(n_partitions + 1)]

    for i in tqdm(range(n_partitions), desc="Repartitioning", unit="partition"):
        chunk_path = output_dir / f"part_{i:04d}.parquet"
        if chunk_path.exists():
            continue

        lo, hi = edges[i], edges[i + 1]

        # last partition is inclusive on right edge
        if i < n_partitions - 1:
            filt = (pl.col("random_index") >= lo) & (pl.col("random_index") < hi)
        else:
            filt = (pl.col("random_index") >= lo) & (pl.col("random_index") <= hi)

        (
            pl.scan_parquet(files)
            .filter(filt)
            .sink_parquet(
                chunk_path,
                compression="zstd",
                row_group_size=100_000,
            )
        )

    print("Done.")

if __name__ == "__main__":
    INPUT_PATTERN = "/shares/soares-santos.physik.uzh/euclid/bucketsx8_with_w/*.parquet"
    OUTPUT_FOLDER = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8_with_w/small_files")
    repartition_files(INPUT_PATTERN, output_dir=OUTPUT_FOLDER, n_partitions=100)