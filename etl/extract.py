import pandas as pd
import os

DEFAULT_RAW_DIR = r"D:\Desktop\project\data\raw"

FILES = [
    "patients.csv.gz",
    "admissions.csv.gz",
    "diagnoses_icd.csv.gz",
    "labevents.csv.gz",
    "d_icd_diagnoses.csv.gz",
    "d_labitems.csv.gz",
]

def load_all(raw_dir=DEFAULT_RAW_DIR):
    """Load all source files and return dict: {TableName: DataFrame}"""
    tables = {}
    for fname in FILES:
        path = os.path.join(raw_dir, fname)
        table_name = fname.replace(".csv.gz", "")
        df = pd.read_csv(path, compression="gzip", low_memory=False)
        tables[table_name] = df
        print(f"[Extract] {table_name}: {len(df)} rows, {len(df.columns)} cols")
    return tables

if __name__ == "__main__":
    tables = load_all()
    print("\nAll files have finished loading!")