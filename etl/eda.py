"""
Full Data EDA — Chunked version for large files.
Incorporates all deep-dive checks from original demo EDA.
Run from project root.
"""

import os
import pandas as pd
import numpy as np
from config import RAW_DIR, MODE

RAW_FULL         = RAW_DIR
CHUNK_SIZE       = 500_000
MAX_CHUNKS_LARGE = 10    # ~5M rows sampled for labevents
MAX_CHUNKS_MED   = 5     # ~2.5M rows for prescriptions

def fmt(n): return f"{n:,}"

def null_and_empty(series):
    """Return (null_count, empty_string_count) for a series."""
    null_n  = series.isna().sum()
    empty_n = (series.astype(str).str.strip() == "").sum()
    return null_n, empty_n


# ─────────────────────────────────────────────────────────────
# Small files: full load + full deep dive
# ─────────────────────────────────────────────────────────────

def eda_small(filename):
    path = os.path.join(RAW_FULL, filename)
    name = filename.replace(".csv.gz", "")
    df   = pd.read_csv(path, compression="gzip", low_memory=False)

    print(f"\nTABLE: {name.upper()}  |  rows: {fmt(len(df))}  |  cols: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")

    # Null + empty string table
    print(f"\n{'  Column':<35} {'Dtype':<12} {'Nulls':>8} {'Null%':>7}  {'Empty_str':>9}")
    print(f"  {'-'*35} {'-'*12} {'-'*8} {'-'*7}  {'-'*9}")
    for col in df.columns:
        null_n, empty_n = null_and_empty(df[col])
        flag = " *" if null_n > 0 or empty_n > 0 else ""
        print(f"  {col:<35} {str(df[col].dtype):<12} {fmt(null_n):>8} "
              f"{null_n/len(df)*100:>6.1f}%  {fmt(empty_n):>9}{flag}")

    # Deep dive
    print(f"\n[Key Field Details]")
    _deep_dive_small(name, df)

    return df


def _deep_dive_small(name, df):
    N = len(df)

    if name == "patients":
        _vc(df, "gender", N)
        _num_stats(df, "anchor_age")
        dod_null = df["dod"].isna().sum()
        print(f"\n  dod null (alive): {fmt(dod_null)} ({dod_null/N*100:.1f}%)")

    elif name == "admissions":
        _vc(df, "admission_type", N)
        _vc(df, "insurance", N)
        _vc(df, "marital_status", N)
        _vc(df, "race", N, top_n=8)
        print(f"\n  Timestamp null counts:")
        for col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
            null_n, empty_n = null_and_empty(df[col])
            print(f"    {col:<20} null={fmt(null_n)}  empty_str={fmt(empty_n)}")

    elif name == "diagnoses_icd":
        _vc(df, "icd_version", N)
        print(f"\n  Unique icd_code: {df['icd_code'].nunique()}")
        print(f"  Unique (subject_id, hadm_id) pairs: {df.groupby(['subject_id','hadm_id']).ngroups}")
        dup = df.duplicated(subset=["subject_id", "hadm_id", "seq_num"]).sum()
        print(f"  Duplicates on (subject_id, hadm_id, seq_num): {fmt(dup)}")

    elif name == "d_icd_diagnoses":
        _vc(df, "icd_version", N)
        print(f"  Unique icd_code: {df['icd_code'].nunique()}")
        print(f"  long_title null: {df['long_title'].isna().sum()}")
        print(f"\n  ICD-10 hierarchy preview (first 10):")
        for _, row in df[df["icd_version"] == 10].head(10).iterrows():
            code   = str(row["icd_code"])
            parent = code[:3] if len(code) > 3 else code
            print(f"    {code} → parent: {parent} | {str(row['long_title'])[:60]}")
        print(f"\n  ICD-9 hierarchy preview (first 5):")
        for _, row in df[df["icd_version"] == 9].head(5).iterrows():
            code   = str(row["icd_code"])
            parent = code[:3] if len(code) > 3 else code
            print(f"    {code} → parent: {parent} | {str(row['long_title'])[:60]}")

    elif name == "d_labitems":
        _vc(df, "category", N)
        _vc(df, "fluid", N)
        print(f"\n  Unique itemid: {df['itemid'].nunique()}")
        print(f"  label null: {df['label'].isna().sum()}")


def _vc(df, col, N, top_n=5):
    print(f"\n  {col} distribution (top {top_n}):")
    for val, cnt in df[col].value_counts(dropna=False).head(top_n).items():
        display = repr(val) if pd.isna(val) or val == "" else val
        print(f"    {display}: {fmt(cnt)} ({cnt/N*100:.1f}%)")


def _num_stats(df, col):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s):
        print(f"\n  {col}: min={s.min():.1f}  25%={s.quantile(.25):.1f}  "
              f"median={s.median():.1f}  75%={s.quantile(.75):.1f}  "
              f"max={s.max():.1f}  mean={s.mean():.1f}")


# ─────────────────────────────────────────────────────────────
# Large files: chunked
# ─────────────────────────────────────────────────────────────

def eda_labevents(max_chunks=MAX_CHUNKS_LARGE):
    path = os.path.join(RAW_FULL, "labevents.csv.gz")
    print(f"\nTABLE: LABEVENTS  (chunked, up to {fmt(max_chunks * CHUNK_SIZE)} rows sampled)")

    total = 0
    # null + empty string counters
    null_counts  = {c: 0 for c in ["hadm_id","valuenum","ref_range_lower","ref_range_upper","value","valueuom"]}
    empty_counts = {c: 0 for c in ["value","valueuom"]}
    both_present = 0          # ref_range_lower AND upper both non-null
    abnormal_cnt = 0
    vnum_min, vnum_max, vnum_sum, vnum_n = np.inf, -np.inf, 0.0, 0
    chunks_read  = 0

    reader = pd.read_csv(path, compression="gzip", low_memory=False, chunksize=CHUNK_SIZE)
    for chunk in reader:
        if chunks_read == 0:
            print(f"Columns: {list(chunk.columns)}")
        total += len(chunk)

        for c in null_counts:
            if c in chunk.columns:
                null_counts[c] += chunk[c].isna().sum()
        for c in empty_counts:
            if c in chunk.columns:
                empty_counts[c] += (chunk[c].astype(str).str.strip() == "").sum()

        mask = chunk["ref_range_lower"].notna() & chunk["ref_range_upper"].notna()
        both_present += mask.sum()

        if "flag" in chunk.columns:
            abnormal_cnt += (chunk["flag"] == "abnormal").sum()

        valid_vnum = pd.to_numeric(chunk["valuenum"], errors="coerce").dropna()
        if len(valid_vnum):
            vnum_min  = min(vnum_min, valid_vnum.min())
            vnum_max  = max(vnum_max, valid_vnum.max())
            vnum_sum += valid_vnum.sum()
            vnum_n   += len(valid_vnum)

        chunks_read += 1
        if chunks_read >= max_chunks:
            break

    print(f"\nRows sampled: {fmt(total)}  (first {chunks_read} chunks)")

    print(f"\nNull / empty rates:")
    print(f"  {'Field':<30} {'Nulls':>10} {'Null%':>7}  {'Empty_str':>10}")
    for c in null_counts:
        ec    = empty_counts.get(c, 0)
        null_n = null_counts[c]
        print(f"  {c:<30} {fmt(null_n):>10} {null_n/total*100:>6.1f}%  {fmt(ec):>10}")

    print(f"\n  Both ref ranges present: {fmt(both_present)} ({both_present/total*100:.1f}%)")
    print(f"  Flagged 'abnormal':      {fmt(abnormal_cnt)} ({abnormal_cnt/total*100:.1f}%)")

    vnum_mean = vnum_sum / vnum_n if vnum_n else float("nan")
    print(f"\n  valuenum: min={vnum_min:.2f}  max={vnum_max:.2f}  mean={vnum_mean:.2f}  "
          f"valid_n={fmt(vnum_n)}")

    print(f"\n  [NOTE] hadm_id null = {null_counts['hadm_id']/total*100:.1f}% "
          f"— these rows cannot link to Admission node")


def eda_prescriptions(max_chunks=MAX_CHUNKS_MED):
    path = os.path.join(RAW_FULL, "prescriptions.csv.gz")
    print(f"\nTABLE: PRESCRIPTIONS  (chunked, up to {fmt(max_chunks * CHUNK_SIZE)} rows sampled)")

    total = 0
    null_counts  = {c: 0 for c in ["hadm_id","drug","dose_val_rx","dose_unit_rx","route"]}
    empty_counts = {c: 0 for c in ["hadm_id","drug","dose_val_rx","dose_unit_rx","route"]}
    drug_type_vc = {}
    top_drugs    = {}
    chunks_read  = 0

    reader = pd.read_csv(path, compression="gzip", low_memory=False, chunksize=CHUNK_SIZE)
    for chunk in reader:
        if chunks_read == 0:
            print(f"Columns: {list(chunk.columns)}")
        total += len(chunk)

        for c in null_counts:
            if c in chunk.columns:
                null_counts[c]  += chunk[c].isna().sum()
                empty_counts[c] += (chunk[c].astype(str).str.strip() == "").sum()

        if "drug_type" in chunk.columns:
            for k, v in chunk["drug_type"].value_counts(dropna=False).items():
                drug_type_vc[k] = drug_type_vc.get(k, 0) + v

        if "drug" in chunk.columns:
            for k, v in chunk["drug"].value_counts().head(20).items():
                top_drugs[k] = top_drugs.get(k, 0) + v

        chunks_read += 1
        if chunks_read >= max_chunks:
            break

    print(f"\nRows sampled: {fmt(total)}  (first {chunks_read} chunks)")

    print(f"\nNull / empty rates:")
    print(f"  {'Field':<25} {'Nulls':>10} {'Null%':>7}  {'Empty_str':>10}")
    for c in null_counts:
        null_n = null_counts[c]
        emp_n  = empty_counts[c]
        print(f"  {c:<25} {fmt(null_n):>10} {null_n/total*100:>6.1f}%  {fmt(emp_n):>10}")

    print(f"\n  drug_type distribution:")
    for k, v in sorted(drug_type_vc.items(), key=lambda x: -x[1]):
        print(f"    {k}: {fmt(v)} ({v/total*100:.1f}%)")

    print(f"\n  Top 10 drugs (in sampled rows):")
    for drug, cnt in sorted(top_drugs.items(), key=lambda x: -x[1])[:10]:
        print(f"    {drug}: {fmt(cnt)}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"FULL DATA EDA — MIMIC-IV  |  MODE: {MODE.upper()}  |  Source: {RAW_DIR}")

    eda_small("patients.csv.gz")
    eda_small("admissions.csv.gz")
    eda_small("diagnoses_icd.csv.gz")
    eda_small("d_icd_diagnoses.csv.gz")
    eda_small("d_labitems.csv.gz")
    eda_labevents()
    eda_prescriptions()

    print("\nEDA COMPLETE")