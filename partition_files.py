import polars as pl
from pathlib import Path
from tqdm import tqdm
import glob

def repartition_files(
    input_glob:   str,
    output_dir:   Path,
    n_partitions: int = 100,
):
    tmp_dir    = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(input_glob))
    print(f"Found {len(files)} input files → {n_partitions} partitions")

    edges = [i / n_partitions for i in range(n_partitions + 1)]

    # ── pass 1: read each file ONCE, scatter to all partitions ───────────────
    for file in tqdm(files, desc="Pass 1 (scatter)", unit="file"):
        stem = Path(file).stem

        # check if all fragments for this file already exist
        if all((tmp_dir / f"part_{i:04d}_{stem}.parquet").exists()
               for i in range(n_partitions)):
            print(f"  {stem} already scattered, skipping")
            continue

        # load entire file once
        df = pl.read_parquet(file)

        for i in range(n_partitions):
            frag_path = tmp_dir / f"part_{i:04d}_{stem}.parquet"
            if frag_path.exists():
                continue
            lo, hi = edges[i], edges[i + 1]
            mask = (df["random_index"] >= lo) & (
                df["random_index"] <= hi if i == n_partitions - 1
                else df["random_index"] < hi
            )
            df.filter(mask).write_parquet(
                frag_path, compression="zstd", row_group_size=100_000
            )

        del df  # free RAM before next file

    # ── pass 2: gather + sort fragments into final files ─────────────────────
    for i in tqdm(range(n_partitions), desc="Pass 2 (gather)", unit="partition"):
        out_path = output_dir / f"part_{i:04d}.parquet"
        if out_path.exists():
            continue
        frags = sorted(tmp_dir.glob(f"part_{i:04d}_*.parquet"))
        if not frags:
            continue
        (
            pl.read_parquet(frags)
            .sort("random_index")
            .write_parquet(out_path, compression="zstd", row_group_size=100_000)
        )

    import shutil
    shutil.rmtree(tmp_dir)
    print("Done.")

if __name__ == "__main__":
    INPUT_PATTERN = "/shares/soares-santos.physik.uzh/euclid/bucketsx8_with_w/*.parquet"
    OUTPUT_FOLDER = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8_sorted")
    repartition_files(INPUT_PATTERN, output_dir=OUTPUT_FOLDER, n_partitions=1000)