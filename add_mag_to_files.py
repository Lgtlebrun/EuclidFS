import polars as pl
from pathlib import Path
from tqdm import tqdm
import os
from EuclidFS.data import MAG2FLUX_MAP

# Assuming these are available in your environment
# from your_project.constants import MAG2FLUX_MAP, FLUX_COLUMNS

def bake_magnitudes(folder_path: str):
    folder = Path(folder_path)
    # Get all parts, sorted so the progress bar is predictable
    files = sorted(list(folder.glob("*.parquet")))
    
    if not files:
        print(f"No parquet files found in {folder_path}")
        return

    print(f"Starting magnitude bake for {len(files)} files...")

    # Initialize the progress bar
    pbar = tqdm(files, desc="Baking Magnitudes", unit="file")
    
    for f_path in pbar:
        pbar.set_postfix(file=f_path.name[:20])
        
        # 1. Scan metadata only to check if we need to do work
        lf = pl.scan_parquet(f_path)
        existing_cols = lf.collect_schema().names()
        
        # Determine which magnitudes are missing
        mag_exprs = [
            pl.when(pl.col(flux_col) > 0)
              .then(-2.5 * pl.col(flux_col).log(10) - 48.6)
              .otherwise(None)
              .alias(mag_col)
            for mag_col, flux_col in MAG2FLUX_MAP.items()
            if mag_col not in existing_cols
        ]

        # If all magnitude columns already exist, skip this file
        if not mag_exprs:
            continue

        # 2. Compute and Write
        tmp_path = f_path.with_suffix(".tmp")
        try:
            (
                lf.with_columns(mag_exprs)
                .collect(streaming=True)
                .write_parquet(
                    tmp_path, 
                    compression="zstd", 
                    row_group_size=100_000
                )
            )
            
            # 3. Atomic swap: replace original with complete file
            os.replace(tmp_path, f_path)
            
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink() # Clean up failed attempt
            print(f"\n❌ Error processing {f_path.name}: {e}")
            continue

    print("\n✅ All files processed successfully.")

if __name__ == "__main__":
    # Point this to your new 'small_files' directory
    PATH = "/shares/soares-santos.physik.uzh/euclid/bucketsx8_with_w/small_files"
    bake_magnitudes(PATH)