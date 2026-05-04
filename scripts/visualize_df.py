import polars as pl
from pathlib import Path
import sys

path_to_data = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8_with_w")

files = [path_to_data / f"{nb}.parquet" for nb in range(23070, 23078)]


if __name__ == "__main__":
	file_nb = sys.argv[1]
	paths = [p for p in files if file_nb in str(p)]
	path = paths[0]
	

	print(f"Reading data in {path}")
	df = pl.scan_parquet(path)
	print(print(df.schema))
	print(df.head(5).collect())
