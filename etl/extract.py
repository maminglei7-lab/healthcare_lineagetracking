"""
Extract: Load all source CSV files from raw directory.
Reads from config.py for paths, file list, and chunked file settings.

Two loading modes:
  - load_all()      : full in-memory load for small files
  - load_chunked()  : chunk generator for large files (defined in config.CHUNKED_FILES)
"""

import pandas as pd
import os
from config import RAW_DIR, SOURCE_FILES, CHUNKED_FILES, CHUNK_SIZE


def load_all(raw_dir=RAW_DIR):
    """
    Load small files into memory. Skips files listed in CHUNKED_FILES.
    Returns dict: {table_name: DataFrame}
    """
    tables = {}
    for fname in SOURCE_FILES:
        if fname in CHUNKED_FILES:
            print(f"[Extract] {fname.replace('.csv.gz','')} → skipped (chunked mode, use load_chunked)")
            continue
        path = os.path.join(raw_dir, fname)
        table_name = fname.replace(".csv.gz", "")
        df = pd.read_csv(path, compression="gzip", low_memory=False)
        tables[table_name] = df
        print(f"[Extract] {table_name}: {len(df):,} rows, {len(df.columns)} cols")
    return tables


def load_chunked(fname, raw_dir=RAW_DIR):
    """
    Generator: yield one chunk at a time for large files.
    Usage:
        for chunk in load_chunked("labevents.csv.gz"):
            ...
    """
    path = os.path.join(raw_dir, fname)
    table_name = fname.replace(".csv.gz", "")
    print(f"[Extract] {table_name}: reading in chunks of {CHUNK_SIZE:,} rows")
    reader = pd.read_csv(path, compression="gzip", low_memory=False, chunksize=CHUNK_SIZE)
    for i, chunk in enumerate(reader):
        print(f"  chunk {i+1}: {len(chunk):,} rows")
        yield chunk


if __name__ == "__main__":
    tables = load_all()
    print(f"\nSmall files loaded: {list(tables.keys())}")
    print(f"Chunked files (not loaded here): {[f.replace('.csv.gz','') for f in CHUNKED_FILES]}")