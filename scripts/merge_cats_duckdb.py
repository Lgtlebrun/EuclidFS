import duckdb
from pathlib import Path

BUCKETS_PATH = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8")
EXTRA_PATH   = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8/25697.parquet")
OUT_PATH     = Path("/shares/soares-santos.physik.uzh/euclid/bucketsx8_merged")
OUT_PATH.mkdir(exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='350GB'")
con.execute("SET temp_directory='~/scratch/duckdb_spill'")  # spills to disk if needed

for i in range(23070, 23078):
    print(f"Merging bucket {i}...")
    con.execute(f"""
        COPY (
            SELECT b.*, e.* EXCLUDE (galaxy_id, halo_id)
            FROM read_parquet('{BUCKETS_PATH}/{i}.parquet') b
            LEFT JOIN read_parquet('{EXTRA_PATH}') e
              ON b.galaxy_id = e.galaxy_id AND b.halo_id = e.halo_id
        )
        TO '{OUT_PATH}/{i}.parquet' (FORMAT parquet, COMPRESSION zstd)
    """)
    print(f"  bucket {i} done")
