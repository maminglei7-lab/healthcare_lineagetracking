import sys
sys.path.append(r"D:\Desktop\project\etl")

import pandas as pd
from lineage_decorator import capture_lineage

# ─────────────────────────────────────────────
# PATIENTS
# ─────────────────────────────────────────────

@capture_lineage(
    sources=["patients.dod"],
    target="patients.dod",
    transformation="fill_missing",
    description="Fill missing dod values with 'unknown' to indicate no recorded death date"
)
def clean_patients_dod(df):
    df = df.copy()
    df["dod"] = df["dod"].fillna("unknown")
    return df

def transform_patients(df):
    print(f"[Transform] patients Start: {len(df)} rows")
    df = clean_patients_dod(df)
    before = len(df)
    df = df.drop_duplicates(subset=["subject_id"])
    print(f"[Transform] patients Deduplication: {before} → {len(df)} rows")
    print(f"[Transform] patients Completed")
    return df


# ─────────────────────────────────────────────
# ADMISSIONS
# ─────────────────────────────────────────────

@capture_lineage(
    sources=["admissions.deathtime", "admissions.edregtime", "admissions.edouttime"],
    target="admissions.deathtime,admissions.edregtime,admissions.edouttime",
    transformation="fill_missing",
    description="Fill missing optional time fields (deathtime, edregtime, edouttime) with 'unknown'"
)
def clean_admissions_nulls(df):
    df = df.copy()
    for col in ["deathtime", "edregtime", "edouttime"]:
        df[col] = df[col].fillna("unknown")
    return df

@capture_lineage(
    sources=["admissions.language"],
    target="admissions.language",
    transformation="replace_unknown",
    description="Replace '?' values in language field with 'unknown'"
)
def clean_admissions_language(df):
    df = df.copy()
    df["language"] = df["language"].replace("?", "unknown")
    return df

@capture_lineage(
    sources=["admissions.marital_status"],
    target="admissions.marital_status",
    transformation="fill_missing",
    description="Fill missing marital_status values with 'unknown'"
)
def clean_admissions_marital(df):
    df = df.copy()
    df["marital_status"] = df["marital_status"].fillna("unknown")
    return df

def transform_admissions(df):
    print(f"[Transform] admissions Start: {len(df)} rows")
    df = clean_admissions_nulls(df)
    df = clean_admissions_language(df)
    df = clean_admissions_marital(df)
    before = len(df)
    df = df.drop_duplicates(subset=["hadm_id"])
    print(f"[Transform] admissions Deduplication: {before} → {len(df)} rows")
    print(f"[Transform] admissions Completed")
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
    df["diagnosis_id"] = df["icd_code"] + "_v" + df["icd_version"].astype(str)
    return df

def transform_diagnoses(df):
    print(f"[Transform] diagnoses_icd Start: {len(df)} rows")
    df = add_diagnosis_id(df)
    before = len(df)
    df = df.drop_duplicates(subset=["subject_id", "hadm_id", "seq_num"])
    print(f"[Transform] diagnoses_icd Deduplication: {before} → {len(df)} rows")
    print(f"[Transform] diagnoses_icd Completed")
    return df


# ─────────────────────────────────────────────
# LABEVENTS
# ─────────────────────────────────────────────

@capture_lineage(
    sources=["labevents.valuenum"],
    target="labevents.valuenum",
    transformation="fill_missing_median",
    description="Fill missing valuenum with per-itemid median; fallback to global median if entire group is null"
)
def clean_labevents_valuenum(df):
    df = df.copy()
    median_by_item = df.groupby("itemid")["valuenum"].median()
    df["valuenum"] = df["valuenum"].fillna(df["itemid"].map(median_by_item))
    global_median = df["valuenum"].median()
    df["valuenum"] = df["valuenum"].fillna(global_median)
    return df

@capture_lineage(
    sources=["labevents.flag"],
    target="labevents.flag",
    transformation="fill_missing",
    description="Fill missing flag values with 'normal' to indicate no abnormality recorded"
)
def clean_labevents_flag(df):
    df = df.copy()
    df["flag"] = df["flag"].fillna("normal")
    return df

@capture_lineage(
    sources=["labevents.hadm_id"],
    target="labevents.hadm_id",
    transformation="fill_missing",
    description="Fill missing hadm_id with -1 to indicate lab result not linked to any admission"
)
def clean_labevents_hadm(df):
    df = df.copy()
    df["hadm_id"] = df["hadm_id"].fillna(-1).astype(int)
    return df

def transform_labevents(df):
    print(f"[Transform] labevents Start: {len(df)} rows")
    df = clean_labevents_valuenum(df)
    df = clean_labevents_flag(df)
    df = clean_labevents_hadm(df)
    before = len(df)
    df = df.drop_duplicates(subset=["labevent_id"])
    print(f"[Transform] labevents Deduplication: {before} → {len(df)} rows")
    print(f"[Transform] labevents Completed")
    return df


# ─────────────────────────────────────────────
# Dictionary table
# ─────────────────────────────────────────────

def transform_d_icd_diagnoses(df):
    print(f"[Transform] d_icd_diagnoses: {len(df)} rows, Direct pass-through of dictionary tables")
    return df

def transform_d_labitems(df):
    df = df.copy()
    df["label"] = df["label"].fillna("unknown")
    print(f"[Transform] d_labitems: {len(df)} rows, label Null values are filled with 'unknown'")
    return df


# ─────────────────────────────────────────────
# Unified Portal
# ─────────────────────────────────────────────

def transform_all(tables):
    return {
        "patients":        transform_patients(tables["patients"]),
        "admissions":      transform_admissions(tables["admissions"]),
        "diagnoses_icd":   transform_diagnoses(tables["diagnoses_icd"]),
        "labevents":       transform_labevents(tables["labevents"]),
        "d_icd_diagnoses": transform_d_icd_diagnoses(tables["d_icd_diagnoses"]),
        "d_labitems":      transform_d_labitems(tables["d_labitems"]),
    }