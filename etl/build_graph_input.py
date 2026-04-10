"""
Build Graph Input: Generate Neo4j-ready CSV files for nodes and relationships.

Graph Schema:
  Layer 1 — Clinical Entity Layer:
    Patient → Admission → Diagnosis / Medication / LabTest
  Layer 2 — Knowledge Enhancement Layer:
    Diagnosis -[BELONGS_TO_CATEGORY]→ ICD_Category
    LabTest ref_range stored on HAS_LAB_RESULT relationship

Large files (labevents 84.6M, prescriptions 20.3M) are processed in chunks
to avoid OOM. CHUNK_SIZE is read from config.
"""

import pandas as pd
import os
from config import CLEANED_DIR, GRAPH_INPUT_DIR, NODES_DIR, RELS_DIR, CHUNK_SIZE


def makedirs():
    os.makedirs(NODES_DIR, exist_ok=True)
    os.makedirs(RELS_DIR, exist_ok=True)


def save(df, path):
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[Save] {os.path.relpath(path, GRAPH_INPUT_DIR)} → {len(df):,} rows")


def read_cleaned(table_name):
    return pd.read_csv(
        os.path.join(CLEANED_DIR, f"{table_name}.csv"), low_memory=False
    )


def read_cleaned_chunked(table_name):
    """Generator: yield chunks from a large cleaned CSV."""
    return pd.read_csv(
        os.path.join(CLEANED_DIR, f"{table_name}.csv"),
        low_memory=False,
        chunksize=CHUNK_SIZE,
    )


def append_csv(df, path, first_chunk):
    """Append a chunk to a CSV; write header on first chunk only."""
    df.to_csv(path, index=False, encoding="utf-8",
              mode="a", header=first_chunk)


# ═════════════════════════════════════════════
#  LAYER 1: CLINICAL ENTITY NODES
# ═════════════════════════════════════════════

def build_patients():
    df = read_cleaned("patients")
    out = pd.DataFrame({
        "patientId:ID": "p_" + df["subject_id"].astype(str),
        "subjectId":    df["subject_id"].astype(int),
        "gender":       df["gender"],
        "anchorAge":    df["anchor_age"].astype(int),
        "anchorYear":   df["anchor_year"].astype(int),
        ":LABEL":       "Patient",
    })
    save(out, os.path.join(NODES_DIR, "patients.csv"))


def build_admissions():
    df = read_cleaned("admissions")
    out = pd.DataFrame({
        "admissionId:ID": "a_" + df["hadm_id"].astype(str),
        "hadmId":         df["hadm_id"].astype(int),
        "admissionType":  df["admission_type"],
        "admitTime":      df["admittime"].fillna(""),
        "dischargeTime":  df["dischtime"].fillna(""),
        "insurance":      df["insurance"],
        "language":       df["language"],
        "maritalStatus":  df["marital_status"],
        "race":           df["race"],
        ":LABEL":         "Admission",
    })
    save(out, os.path.join(NODES_DIR, "admissions.csv"))


def build_diagnoses():
    diag = read_cleaned("diagnoses_icd")
    desc = read_cleaned("d_icd_diagnoses")
    unique = diag[["icd_code", "icd_version"]].drop_duplicates()
    merged = unique.merge(desc, on=["icd_code", "icd_version"], how="left")
    out = pd.DataFrame({
        "diagnosisId:ID": "d_" + merged["icd_code"] + "_" + merged["icd_version"].astype(str),
        "icdCode":        merged["icd_code"],
        "icdVersion":     merged["icd_version"].astype(int),
        "icdTitle":       merged["long_title"].fillna("Unknown"),
        ":LABEL":         "Diagnosis",
    })
    save(out, os.path.join(NODES_DIR, "diagnoses.csv"))


def build_labtests():
    df = read_cleaned("d_labitems")
    out = pd.DataFrame({
        "labTestId:ID": "lt_" + df["itemid"].astype(str),
        "itemId":       df["itemid"].astype(int),
        "label":        df["label"],
        "fluid":        df["fluid"],
        "category":     df["category"],
        ":LABEL":       "LabTest",
    })
    save(out, os.path.join(NODES_DIR, "labtests.csv"))


def build_medications():
    """
    Medication nodes — one node per unique drug name.
    Prescriptions is 20.3M rows, so we accumulate unique drug names
    across chunks before building nodes.
    Dose/route/time go on HAS_PRESCRIPTION relationship, not here.
    """
    print("[Build] medications — collecting unique drug names (chunked)...")
    unique_drugs = set()
    for chunk in read_cleaned_chunked("prescriptions"):
        unique_drugs.update(chunk["drug"].dropna().unique().tolist())

    unique_drugs = sorted(unique_drugs)
    print(f"  Unique drug names found: {len(unique_drugs):,}")

    out = pd.DataFrame({
        "medicationId:ID": ["med_" + str(i) for i in range(len(unique_drugs))],
        "drugName":        unique_drugs,
        ":LABEL":          "Medication",
    })

    # Save lookup for relationship building
    lookup = pd.DataFrame({
        "drug":         unique_drugs,
        "medicationId": ["med_" + str(i) for i in range(len(unique_drugs))],
    })
    lookup.to_csv(os.path.join(NODES_DIR, "_medication_lookup.csv"), index=False)

    save(out, os.path.join(NODES_DIR, "medications.csv"))


# ═════════════════════════════════════════════
#  LAYER 2: KNOWLEDGE ENHANCEMENT NODES
# ═════════════════════════════════════════════

def build_icd_categories():
    diag = read_cleaned("diagnoses_icd")
    desc = read_cleaned("d_icd_diagnoses")

    unique_codes = diag[["icd_code", "icd_version"]].drop_duplicates().copy()
    unique_codes["parent_code"] = unique_codes["icd_code"].str[:3]
    parents = unique_codes[["parent_code", "icd_version"]].drop_duplicates().reset_index(drop=True)

    parent_titles = []
    for _, row in parents.iterrows():
        p_code, p_ver = row["parent_code"], row["icd_version"]
        exact = desc[(desc["icd_code"] == p_code) & (desc["icd_version"] == p_ver)]
        if len(exact) > 0:
            parent_titles.append(exact.iloc[0]["long_title"])
        else:
            children = desc[
                desc["icd_code"].str.startswith(p_code) & (desc["icd_version"] == p_ver)
            ]
            parent_titles.append(children.iloc[0]["long_title"] if len(children) > 0 else "Unknown")

    parents["categoryTitle"] = parent_titles
    out = pd.DataFrame({
        "categoryId:ID": "icd_" + parents["parent_code"] + "_v" + parents["icd_version"].astype(str),
        "code":          parents["parent_code"],
        "icdVersion":    parents["icd_version"].astype(int),
        "title":         parents["categoryTitle"],
        ":LABEL":        "ICD_Category",
    })
    save(out, os.path.join(NODES_DIR, "icd_categories.csv"))


# ═════════════════════════════════════════════
#  LAYER 1: RELATIONSHIPS
# ═════════════════════════════════════════════

def build_had_admission():
    df = read_cleaned("admissions")
    out = pd.DataFrame({
        ":START_ID": "p_" + df["subject_id"].astype(str),
        ":END_ID":   "a_" + df["hadm_id"].astype(str),
        ":TYPE":     "HAD_ADMISSION",
    })
    save(out, os.path.join(RELS_DIR, "had_admission.csv"))


def build_has_diagnosis():
    df = read_cleaned("diagnoses_icd")
    out = pd.DataFrame({
        ":START_ID": "a_" + df["hadm_id"].astype(str),
        ":END_ID":   "d_" + df["icd_code"] + "_" + df["icd_version"].astype(str),
        ":TYPE":     "HAS_DIAGNOSIS",
        "seqNum":    df["seq_num"].astype(int),
    })
    save(out, os.path.join(RELS_DIR, "has_diagnosis.csv"))


def build_has_lab_result():
    """
    Admission → LabTest relationship.
    labevents cleaned is 84.6M rows — processed in chunks, appended to CSV.
    ref_range on relationship enables abnormal detection queries.
    """
    out_path = os.path.join(RELS_DIR, "has_lab_result.csv")
    if os.path.exists(out_path):
        os.remove(out_path)

    total = 0
    first_chunk = True
    for chunk in read_cleaned_chunked("labevents"):
        out = pd.DataFrame({
            ":START_ID":     "a_" + chunk["hadm_id"].astype(int).astype(str),
            ":END_ID":       "lt_" + chunk["itemid"].astype(str),
            ":TYPE":         "HAS_LAB_RESULT",
            "value":         chunk["valuenum"],
            "flag":          chunk["flag"],
            "chartTime":     chunk["charttime"].fillna(""),
            "refRangeLower": chunk["ref_range_lower"],
            "refRangeUpper": chunk["ref_range_upper"],
            "valueuom":      chunk["valueuom"].fillna(""),
            "outlier":       chunk["valuenum_outlier"],
        })
        append_csv(out, out_path, first_chunk)
        total += len(out)
        first_chunk = False
        print(f"  has_lab_result: {total:,} rows written...", end="\r")

    print(f"\n[Save] relationships/has_lab_result.csv → {total:,} rows")


def build_has_prescription():
    """
    Admission → Medication relationship.
    prescriptions cleaned is 20.3M rows — processed in chunks.
    Medication lookup loaded once into memory (unique drugs ~tens of thousands).
    """
    lookup = pd.read_csv(os.path.join(NODES_DIR, "_medication_lookup.csv"))

    out_path = os.path.join(RELS_DIR, "has_prescription.csv")
    if os.path.exists(out_path):
        os.remove(out_path)

    total = 0
    first_chunk = True
    for chunk in read_cleaned_chunked("prescriptions"):
        merged = chunk.merge(lookup, on="drug", how="left")
        out = pd.DataFrame({
            ":START_ID": "a_" + merged["hadm_id"].astype(int).astype(str),
            ":END_ID":   merged["medicationId"],
            ":TYPE":     "HAS_PRESCRIPTION",
            "drugType":  merged["drug_type"],
            "doseVal":   merged["dose_val_rx"].fillna(""),
            "doseUnit":  merged["dose_unit_rx"].fillna(""),
            "route":     merged["route"],
            "startTime": merged["starttime"].fillna(""),
            "stopTime":  merged["stoptime"].fillna(""),
        })
        append_csv(out, out_path, first_chunk)
        total += len(out)
        first_chunk = False
        print(f"  has_prescription: {total:,} rows written...", end="\r")

    print(f"\n[Save] relationships/has_prescription.csv → {total:,} rows")


# ═════════════════════════════════════════════
#  LAYER 2: RELATIONSHIPS
# ═════════════════════════════════════════════

def build_belongs_to_category():
    diag = read_cleaned("diagnoses_icd")
    unique = diag[["icd_code", "icd_version"]].drop_duplicates().copy()
    unique["parent_code"] = unique["icd_code"].str[:3]
    out = pd.DataFrame({
        ":START_ID": "d_" + unique["icd_code"] + "_" + unique["icd_version"].astype(str),
        ":END_ID":   "icd_" + unique["parent_code"] + "_v" + unique["icd_version"].astype(str),
        ":TYPE":     "BELONGS_TO_CATEGORY",
    })
    save(out, os.path.join(RELS_DIR, "belongs_to_category.csv"))


# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════

if __name__ == "__main__":
    makedirs()

    print("=== Layer 1: Clinical Entity Nodes ===")
    build_patients()
    build_admissions()
    build_diagnoses()
    build_labtests()
    build_medications()        # chunked (unique drug accumulation)

    print("\n=== Layer 2: Knowledge Enhancement Nodes ===")
    build_icd_categories()

    print("\n=== Layer 1: Relationships ===")
    build_had_admission()
    build_has_diagnosis()
    build_has_lab_result()     # chunked (84.6M rows)
    build_has_prescription()   # chunked (20.3M rows)

    print("\n=== Layer 2: Relationships ===")
    build_belongs_to_category()

    print("\n✓ graph_input build complete")
    print(f"  Nodes: {NODES_DIR}")
    print(f"  Rels:  {RELS_DIR}")