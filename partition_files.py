import polars as pl
from pathlib import Path
from tqdm import tqdm
import glob

def repartition_files(
    input_glob:    str,
    output_dir:    Path,
    rows_per_file: int = 5_000_000,
    sort_by_random_index: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(input_glob))
    if not files:
        print(f"No files found matching: {input_glob}")
        return
    print(f"Found {len(files)} files to process.")

    # scan all files at once — Polars handles multi-file scans efficiently
    lf = pl.scan_parquet(files)

    if sort_by_random_index:
        print("Sorting by random_index for fast future sampling...")
        lf = lf.sort("random_index")

    total_rows = lf.select(pl.len()).collect().item()
    num_chunks = (total_rows + rows_per_file - 1) // rows_per_file
    print(f"Total rows: {total_rows:,}  →  {num_chunks} output files")

    for i in tqdm(range(num_chunks), desc="Repartitioning", unit="file"):
        chunk_path = output_dir / f"part_{i:04d}.parquet"
        if chunk_path.exists():
            continue
        lf.slice(i * rows_per_file, rows_per_file).sink_parquet(
            chunk_path,
            compression="zstd",
            row_group_size=100_000,
        )

    print("Done.")

if __name__ == "__main__":
    # Ensure this is the glob string
    INPUT_PATTERN = "/shares/soares-santos.physik.uzh/euclid/bucketsx8_with_w/*.parquet"
    OUTPUT_FOLDER = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8_with_w/small_files")

    repartition_files(INPUT_PATTERN, output_dir=OUTPUT_FOLDER)