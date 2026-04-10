"""
ETL Main Entry Point
Pipeline: Extract → Transform → Save → Quality Check

All paths read from config.py — switch MODE there to toggle demo/full.

Two processing paths:
  - Small files : load_all → transform_all → save_cleaned  (in-memory)
  - Large files : load_chunked → transform chunk → append to CSV  (streaming)
  Large files are defined in config.CHUNKED_FILES.
"""

import os
import json
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (MODE, RAW_DIR, CLEANED_DIR, LINEAGE_DIR, LINEAGE_PATH,
                    CHUNKED_FILES, CHUNK_SIZE, print_config)
from extract import load_all, load_chunked
from transform import (transform_all,
                       transform_labevents,
                       transform_prescriptions)
from quality_check import run_quality_checks


# Map chunked file → its transform function
CHUNKED_TRANSFORMS = {
    "labevents.csv.gz":     transform_labevents,
    "prescriptions.csv.gz": transform_prescriptions,
}


def reset_lineage():
    """Clear lineage.json before each run to prevent duplicate accumulation."""
    os.makedirs(LINEAGE_DIR, exist_ok=True)
    with open(LINEAGE_PATH, 'w', encoding='utf-8') as f:
        json.dump({"lineage_records": []}, f)
    print(f"[Lineage] Reset: {LINEAGE_PATH}")


def save_cleaned(tables):
    """Save all cleaned DataFrames to CSV (small files)."""
    os.makedirs(CLEANED_DIR, exist_ok=True)
    for name, df in tables.items():
        out_path = os.path.join(CLEANED_DIR, f"{name}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[Save] {name}.csv → {len(df):,} rows")


def process_chunked_files():
    """
    Stream each large file chunk by chunk:
      read chunk → transform → append to output CSV → discard chunk from memory.
    Header written on first chunk only; subsequent chunks appended without header.
    """
    os.makedirs(CLEANED_DIR, exist_ok=True)

    for fname in CHUNKED_FILES:
        table_name  = fname.replace(".csv.gz", "")
        transform_fn = CHUNKED_TRANSFORMS[fname]
        out_path    = os.path.join(CLEANED_DIR, f"{table_name}.csv")

        # Clear any existing output file before starting
        if os.path.exists(out_path):
            os.remove(out_path)

        total_in = 0
        total_out = 0
        first_chunk = True

        for chunk in load_chunked(fname, RAW_DIR):
            total_in += len(chunk)
            cleaned_chunk = transform_fn(chunk)
            total_out += len(cleaned_chunk)

            cleaned_chunk.to_csv(
                out_path,
                index=False,
                encoding="utf-8",
                mode="a",                        # append
                header=first_chunk,              # write header on first chunk only
            )
            first_chunk = False

        print(f"[Save] {table_name}.csv → {total_out:,} rows "
              f"(from {total_in:,} raw, {total_in - total_out:,} filtered)")


if __name__ == "__main__":
    # ── Config ──
    print("=" * 50)
    print("  ETL Pipeline")
    print("=" * 50)
    print_config()

    # ── Step 0: Reset lineage ──
    print("\n" + "=" * 50)
    print("  Step 0: Reset Lineage")
    print("=" * 50)
    reset_lineage()

    # ── Step 1: Extract + Transform + Save (small files) ──
    print("\n" + "=" * 50)
    print("  Step 1: Small Files — Extract → Transform → Save")
    print("=" * 50)
    raw     = load_all(RAW_DIR)
    cleaned = transform_all(raw)
    save_cleaned(cleaned)

    # ── Step 2: Extract + Transform + Save (large files, chunked) ──
    print("\n" + "=" * 50)
    print("  Step 2: Large Files — Chunked Extract → Transform → Save")
    print("=" * 50)
    process_chunked_files()

    # ── Step 3: Quality Check ──
    print("\n" + "=" * 50)
    print("  Step 3: Quality Check")
    print("=" * 50)
    all_passed = run_quality_checks(CLEANED_DIR)

    # ── Summary ──
    print("\n" + "=" * 50)
    print("  ETL Complete")
    print("=" * 50)
    print(f"  MODE:     {MODE.upper()}")
    print(f"  Cleaned:  {CLEANED_DIR}")
    print(f"  Lineage:  {LINEAGE_PATH}")
    print(f"  Quality:  {'ALL PASSED' if all_passed else 'HAS FAILURES'}")
    print("=" * 50)