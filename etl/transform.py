"""
Transform: Clean and standardize all tables based on EDA-driven rules.

Cleaning philosophy:
  - Only fill nulls when the semantics are clear (e.g., flag=null means "normal")
  - Preserve null when it carries meaning (e.g., dod=null means "alive")
  - Filter out rows that cannot participate in graph relationships
  - Do NOT fabricate data (e.g., no median imputation for valuenum)
  - Mark outliers in-place; never delete valid clinical values
"""

import pandas as pd
from lineage_decorator import capture_lineage


# ─────────────────────────────────────────────
# PATIENTS
# ─────────────────────────────────────────────

def transform_patients(df):
    """
    Cleaning rules (full data: 364,627 rows):
      - No nulls in core fields (subject_id, gender, anchor_age)
      - dod: 89.5% null → keep null (null = alive, semantically meaningful)
      - Deduplicate on subject_id
    """
    print(f"[Transform] patients: {len(df)} rows")
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=["subject_id"])
    if before != len(df):
        print(f"  Dedup: {before} → {len(df)}")

    print(f"[Transform] patients: done")
    return df


# ─────────────────────────────────────────────
# ADMISSIONS
# ─────────────────────────────────────────────

@capture_lineage(
    sources=["admissions.language"],
    target="admissions.language",
    transformation="replace_unknown",
    description="Replace '?' placeholder in language with 'UNKNOWN'"
)
def clean_admissions_language(df):
    df = df.copy()
    df["language"] = df["language"].replace("?", "UNKNOWN")
    return df

@capture_lineage(
    sources=["admissions.marital_status"],
    target="admissions.marital_status",
    transformation="fill_missing",
    description="Fill null marital_status with 'UNKNOWN'"
)
def clean_admissions_marital(df):
    df = df.copy()
    df["marital_status"] = df["marital_status"].fillna("UNKNOWN")
    return df

@capture_lineage(
    sources=["admissions.discharge_location"],
    target="admissions.discharge_location",
    transformation="fill_missing",
    description="Fill null discharge_location with 'UNKNOWN' (27.4% null in full data)"
)
def clean_admissions_discharge_location(df):
    df = df.copy()
    df["discharge_location"] = df["discharge_location"].fillna("UNKNOWN")
    return df

def transform_admissions(df):
    """
    Cleaning rules (full data: 546,028 rows):
      - language: 0.1% null → fill 'UNKNOWN'
      - marital_status: 2.5% null → fill 'UNKNOWN'
      - discharge_location: 27.4% null → fill 'UNKNOWN'
      - deathtime: 97.8% null → keep null (null = survived, semantically meaningful)
      - edregtime/edouttime: 30.5% null → keep null (not all admissions go through ED)
      - Deduplicate on hadm_id
    """
    print(f"[Transform] admissions: {len(df)} rows")
    df = clean_admissions_language(df)
    df = clean_admissions_marital(df)
    df = clean_admissions_discharge_location(df)

    before = len(df)
    df = df.drop_duplicates(subset=["hadm_id"])
    if before != len(df):
        print(f"  Dedup: {before} → {len(df)}")

    print(f"[Transform] admissions: done")
    return df


# ─────────────────────────────────────────────
# DIAGNOSES_ICD
# ─────────────────────────────────────────────

@capture_lineage(
    sources=["diagnoses_icd.icd_code", "diagnoses_icd.icd_version"],
    target="diagnoses_icd.diagnosis_id",
    transformation="concat_version_suffix",
    description="Generate diagnosis_id = icd_code + '_v' + icd_version to avoid ICD-9/10 code collisions"
)
def add_diagnosis_id(df):
    df = df.copy()
    df["icd_code"] = df["icd_code"].astype(str).str.strip()
    df["diagnosis_id"] = df["icd_code"] + "_v" + df["icd_version"].astype(str)
    return df

def transform_diagnoses(df):
    """
    Cleaning rules (full data: 6,364,488 rows):
      - Zero nulls in all fields
      - 116 duplicates on (subject_id, hadm_id, seq_num) → deduplicate
      - Strip whitespace from icd_code (defensive)
      - Generate diagnosis_id for unique graph node identification
    """
    print(f"[Transform] diagnoses_icd: {len(df)} rows")
    df = add_diagnosis_id(df)

    before = len(df)
    df = df.drop_duplicates(subset=["subject_id", "hadm_id", "seq_num"])
    if before != len(df):
        print(f"  Dedup: {before} → {len(df)} ({before - len(df)} duplicates removed)")

    print(f"[Transform] diagnoses_icd: done")
    return df


# ─────────────────────────────────────────────
# LABEVENTS
# ─────────────────────────────────────────────

@capture_lineage(
    sources=["labevents.flag"],
    target="labevents.flag",
    transformation="fill_missing",
    description="Fill null flag with 'normal' — per MIMIC-IV docs, null means result within normal range"
)
def clean_labevents_flag(df):
    df = df.copy()
    df["flag"] = df["flag"].fillna("normal")
    return df

@capture_lineage(
    sources=["labevents.valueuom"],
    target="labevents.valueuom",
    transformation="fill_missing",
    description="Replace empty string valueuom with None — empty string carries no semantic meaning"
)
def clean_labevents_valueuom(df):
    df = df.copy()
    df["valueuom"] = df["valueuom"].replace("", None)
    df["valueuom"] = df["valueuom"].str.strip().replace("", None)
    return df

@capture_lineage(
    sources=["labevents.valuenum", "labevents.ref_range_lower", "labevents.ref_range_upper"],
    target="labevents.valuenum_outlier",
    transformation="outlier_flag",
    description=(
        "Flag valuenum outliers as valuenum_outlier=True. Rule: "
        "if ref_range exists → flag when valuenum outside [lower - 3*width, upper + 3*width]; "
        "if ref_range absent → flag when valuenum < 0 (negative values clinically invalid for most tests). "
        "Original valuenum is never modified."
    )
)
def flag_labevents_outliers(df):
    df = df.copy()
    outlier = pd.Series(False, index=df.index)

    has_range = df["ref_range_lower"].notna() & df["ref_range_upper"].notna()

    # Rule 1: ref_range present — flag if outside 3× range width
    width = df["ref_range_upper"] - df["ref_range_lower"]
    lower_bound = df["ref_range_lower"] - 3 * width
    upper_bound = df["ref_range_upper"] + 3 * width
    out_of_range = (
        df["valuenum"].notna() &
        has_range &
        ((df["valuenum"] < lower_bound) | (df["valuenum"] > upper_bound))
    )
    outlier |= out_of_range

    # Rule 2: no ref_range — flag negative valuenum
    negative_no_range = (
        df["valuenum"].notna() &
        ~has_range &
        (df["valuenum"] < 0)
    )
    outlier |= negative_no_range

    df["valuenum_outlier"] = outlier
    flagged = outlier.sum()
    print(f"  Outlier flag: {flagged:,} rows marked valuenum_outlier=True")
    return df

def transform_labevents(df):
    """
    Cleaning rules (full data, sampled 5M rows):
      - hadm_id: 46.8% null → FILTER OUT (cannot link to Admission in graph).
        KNOWN LIMITATION: ~half of all labevents discarded. Deliberate decision —
        without hadm_id these rows cannot participate in Patient→Admission→LabTest
        traversals, and inferring hadm_id from charttime proximity risks data corruption.
      - valuenum: 13.7% null → keep null (do NOT impute — fabricated values harm RAG accuracy)
      - valuenum outliers → mark valuenum_outlier=True, original value preserved
      - flag: null → fill 'normal' (MIMIC-IV semantics: null flag = within normal range)
      - valueuom: 26,542 empty strings → replace with None
      - ref_range_lower/upper: 20.1% null → keep null (some tests have no reference range)
      - Deduplicate on labevent_id
    """
    print(f"[Transform] labevents: {len(df)} rows")
    df = df.copy()

    # Filter out rows with no hadm_id
    before_filter = len(df)
    df = df[df["hadm_id"].notna()].copy()
    df["hadm_id"] = df["hadm_id"].astype(int)
    print(f"  Filter hadm_id null: {before_filter:,} → {len(df):,} ({before_filter - len(df):,} removed, "
          f"{(before_filter - len(df)) / before_filter * 100:.1f}% — known limitation, see docstring)")

    df = clean_labevents_flag(df)
    df = clean_labevents_valueuom(df)
    df = flag_labevents_outliers(df)

    before = len(df)
    df = df.drop_duplicates(subset=["labevent_id"])
    if before != len(df):
        print(f"  Dedup: {before} → {len(df)}")

    print(f"[Transform] labevents: done")
    return df


# ─────────────────────────────────────────────
# PRESCRIPTIONS
# ─────────────────────────────────────────────

@capture_lineage(
    sources=["prescriptions.drug"],
    target="prescriptions.drug",
    transformation="replace_unknown",
    description="Standardize drug names: strip whitespace"
)
def clean_prescriptions_drug(df):
    df = df.copy()
    df["drug"] = df["drug"].astype(str).str.strip()
    # drop rows where drug is null or empty — cannot create Medication node without a name
    before = len(df)
    df = df[df["drug"].notna() & (df["drug"].str.strip() != "") & (df["drug"] != "nan")]
    if before != len(df):
        print(f"  Dropped {before - len(df)} rows with null/empty drug name")
    return df

@capture_lineage(
    sources=["prescriptions.route"],
    target="prescriptions.route",
    transformation="fill_missing",
    description="Fill null route with 'UNKNOWN' (0.03% null in full data)"
)
def clean_prescriptions_route(df):
    df = df.copy()
    df["route"] = df["route"].fillna("UNKNOWN")
    return df

def transform_prescriptions(df):
    """
    Cleaning rules (full data, sampled 2.5M rows):
      - hadm_id: 0 null → all rows can link to Admission
      - drug: 0 null → strip whitespace only
      - route: 0.03% null → fill 'UNKNOWN'
      - dose_val_rx/dose_unit_rx: 0.05% null → keep null (doesn't affect core drug queries)
    """
    print(f"[Transform] prescriptions: {len(df)} rows")
    df = clean_prescriptions_drug(df)
    df = clean_prescriptions_route(df)
    print(f"[Transform] prescriptions: done")
    return df


# ─────────────────────────────────────────────
# Dictionary Tables
# ─────────────────────────────────────────────

@capture_lineage(
    sources=["d_labitems.label"],
    target="d_labitems.label",
    transformation="fill_missing",
    description="Fill null and empty string label with 'Unknown' (4 nulls + 1 empty string in full data)"
)
def clean_d_labitems_label(df):
    df = df.copy()
    df["label"] = df["label"].fillna("Unknown")
    df["label"] = df["label"].str.strip().replace("", "Unknown")
    return df

def transform_d_icd_diagnoses(df):
    """Pass-through with defensive strip on icd_code."""
    df = df.copy()
    df["icd_code"] = df["icd_code"].astype(str).str.strip()
    print(f"[Transform] d_icd_diagnoses: {len(df)} rows, icd_code stripped")
    return df

def transform_d_labitems(df):
    """
    Cleaning rules (full data: 1,650 rows):
      - label: 4 nulls + 1 empty string → fill 'Unknown'
    """
    df = clean_d_labitems_label(df)
    print(f"[Transform] d_labitems: {len(df)} rows, label nulls and empty strings filled")
    return df


# ─────────────────────────────────────────────
# Unified Portal
# ─────────────────────────────────────────────

def transform_all(tables):
    """Run transformations for small files only.
    Chunked files (labevents, prescriptions) are handled separately in run_etl.py."""
    return {
        "patients":        transform_patients(tables["patients"]),
        "admissions":      transform_admissions(tables["admissions"]),
        "diagnoses_icd":   transform_diagnoses(tables["diagnoses_icd"]),
        "d_icd_diagnoses": transform_d_icd_diagnoses(tables["d_icd_diagnoses"]),
        "d_labitems":      transform_d_labitems(tables["d_labitems"]),
    }