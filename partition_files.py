import polars as pl
from pathlib import Path
from tqdm import tqdm
import glob

def repartition_files(input_glob: str, output_dir: Path, rows_per_file: int = 5_000_000):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of all matching parquet files
    files = glob.glob(input_glob)
    if not files:
        print(f"No files found matching: {input_glob}")
        return

    print(f"Found {len(files)} heavy files to process.")

    # Outer loop for the files themselves
    for file_path in files:
        file_path = Path(file_path)
        
        # 1. Scan metadata only to get length
        lf = pl.scan_parquet(file_path)
        total_rows = lf.select(pl.len()).collect().item()
        num_chunks = (total_rows // rows_per_file) + 1
        
        # Inner loop with tqdm for chunks
        with tqdm(total=num_chunks, desc=f"💾 {file_path.name}", unit="chunk") as pbar:
            for i in range(num_chunks):
                chunk_name = f"{file_path.stem}_part_{i:03d}.parquet"
                chunk_path = output_dir / chunk_name
                
                # Check if file exists to allow resuming if the job crashes
                if not chunk_path.exists():
                    (
                        lf.slice(i * rows_per_file, rows_per_file)
                        .collect(streaming=True)
                        .write_parquet(
                            chunk_path, 
                            compression="zstd", 
                            row_group_size=100_000
                        )
                    )
                
                pbar.update(1)

if __name__ == "__main__":
    # Ensure this is the glob string
    INPUT_PATTERN = "/shares/soares-santos.physik.uzh/euclid/bucketsx8_with_w/*.parquet"
    OUTPUT_FOLDER = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8_with_w/small_files")

    repartition_files(INPUT_PATTERN, output_dir=OUTPUT_FOLDER)