import sys
sys.path.append(r"D:\Desktop\project\etl")

import pandas as pd
import os
import json

# ─────────────────────────────────────────────
# Configuration: Switch between demo and full data
# ─────────────────────────────────────────────
MODE = "demo"   # Switch to "full" to view actual data

PATHS = {
    "demo": {
        "cleaned":     r"D:\Desktop\project\data\cleaned",
        "graph_input": r"D:\Desktop\project\data\graph_input",
    },
    "full": {
        "cleaned":     r"D:\Desktop\project\data\cleaned_full",
        "graph_input": r"D:\Desktop\project\data\graph_input_full",
    }
}

CLEANED_DIR  = PATHS[MODE]["cleaned"]
GRAPH_DIR    = PATHS[MODE]["graph_input"]
LINEAGE_PATH = r"D:\Desktop\project\lineage\lineage.json"

NODES_DIR = os.path.join(GRAPH_DIR, "nodes")
RELS_DIR  = os.path.join(GRAPH_DIR, "relationships")

def makedirs():
    os.makedirs(NODES_DIR, exist_ok=True)
    os.makedirs(RELS_DIR,  exist_ok=True)

def save(df, path):
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[Save] {os.path.relpath(path, GRAPH_DIR)} → {len(df)} rows")

# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

def build_patients():
    df = pd.read_csv(os.path.join(CLEANED_DIR, "patients.csv"), low_memory=False)
    out = pd.DataFrame({
        "patientId:ID": "p_" + df["subject_id"].astype(str),
        "subjectId":    df["subject_id"].astype(int),
        "gender":       df["gender"].fillna("Unknown"),
        "anchorAge":    df["anchor_age"].fillna(-1).astype(int),
        "anchorYear":   df["anchor_year"].fillna(-1).astype(int),
        ":LABEL":       "Patient"
    })
    save(out, os.path.join(NODES_DIR, "patients.csv"))

def build_admissions():
    df = pd.read_csv(os.path.join(CLEANED_DIR, "admissions.csv"), low_memory=False)
    out = pd.DataFrame({
        "admissionId:ID": "a_" + df["hadm_id"].astype(str),
        "hadmId":         df["hadm_id"].astype(int),
        "admissionType":  df["admission_type"].str.upper().fillna("Unknown"),
        "admitTime":      df["admittime"].fillna(""),
        "dischargeTime":  df["dischtime"].fillna(""),
        "hospital":       "Unknown",   #No hospital field in demodb, filled with 'Unknown'
        ":LABEL":         "Admission"
    })
    save(out, os.path.join(NODES_DIR, "admissions.csv"))

def build_diagnoses():
    diag = pd.read_csv(os.path.join(CLEANED_DIR, "diagnoses_icd.csv"), low_memory=False)
    desc = pd.read_csv(os.path.join(CLEANED_DIR, "d_icd_diagnoses.csv"), low_memory=False)

    # Deduplicate by icd_code + icd_version to obtain unique diagnosis nodes
    unique = diag[["icd_code", "icd_version"]].drop_duplicates()
    merged = unique.merge(desc, on=["icd_code", "icd_version"], how="left")

    out = pd.DataFrame({
        "diagnosisId:ID": "d_" + merged["icd_code"].astype(str) + "_" + merged["icd_version"].astype(str),
        "icdCode":        merged["icd_code"].astype(str).str.strip(),
        "icdVersion":     merged["icd_version"].astype(int),
        "icdTitle":       merged["long_title"].fillna("Unknown"),
        "category":       "Unknown",   # Unclassified field in demodb
        ":LABEL":         "Diagnosis"
    })
    save(out, os.path.join(NODES_DIR, "diagnoses.csv"))

def build_labtests():
    df = pd.read_csv(os.path.join(CLEANED_DIR, "d_labitems.csv"), low_memory=False)
    out = pd.DataFrame({
        "labTestId:ID": "lt_" + df["itemid"].astype(str),
        "itemId":       df["itemid"].astype(int),
        "label":        df["label"].fillna("Unknown"),
        "category":     df["category"].fillna(""),
        "unit":         "",   # No unit field in demodb
        ":LABEL":       "LabTest"
    })
    save(out, os.path.join(NODES_DIR, "labtests.csv"))

def build_fields():
    """Extract all source/target fields from lineage.json to generate fields.csv"""
    with open(LINEAGE_PATH, 'r', encoding='utf-8') as f:
        lineage = json.load(f)

    field_ids = set()
    for rec in lineage["lineage_records"]:
        for fid in rec["source_fields"] + rec["target_fields"]:
            field_ids.add(fid)

    rows = []
    for fid in sorted(field_ids):
        # fid Format: f_{table}_{field}
        parts = fid[2:].split("_", 1)   # Remove the 'f_' prefix, split into a maximum of two segments.
        table = parts[0] if len(parts) > 0 else ""
        field = parts[1] if len(parts) > 1 else ""

        # Determining whether a field is derived
        is_source = field not in ["diagnosis_id"]

        rows.append({
            "fieldId:ID":  fid,
            "fieldName":   field,
            "tableName":   table,
            "dataType":    "",
            "isSource":    str(is_source).lower(),
            "description": "",
            ":LABEL":      "Field"
        })

    out = pd.DataFrame(rows)
    save(out, os.path.join(NODES_DIR, "fields.csv"))

def build_transformations():
    """Generate transformations.csv from lineage.json"""
    with open(LINEAGE_PATH, 'r', encoding='utf-8') as f:
        lineage = json.load(f)

    rows = []
    for rec in lineage["lineage_records"]:
        rows.append({
            "transformationId:ID": rec["transformation_id"],
            "transformType":       rec["transform_type"],
            "logic":               rec["logic"],
            "timestamp":           rec["timestamp"],
            "scriptRef":           rec["script_ref"],
            ":LABEL":              "Transformation"
        })

    out = pd.DataFrame(rows)
    save(out, os.path.join(NODES_DIR, "transformations.csv"))


# ─────────────────────────────────────────────
# RELATIONSHIPS
# ─────────────────────────────────────────────

def build_had_admission():
    df = pd.read_csv(os.path.join(CLEANED_DIR, "admissions.csv"), low_memory=False)
    out = pd.DataFrame({
        ":START_ID": "p_" + df["subject_id"].astype(str),
        ":END_ID":   "a_" + df["hadm_id"].astype(str),
        ":TYPE":     "HAD_ADMISSION"
    })
    save(out, os.path.join(RELS_DIR, "had_admission.csv"))

def build_has_diagnosis():
    df = pd.read_csv(os.path.join(CLEANED_DIR, "diagnoses_icd.csv"), low_memory=False)
    out = pd.DataFrame({
        ":START_ID": "a_" + df["hadm_id"].astype(str),
        ":END_ID":   "d_" + df["icd_code"].astype(str) + "_" + df["icd_version"].astype(str),
        ":TYPE":     "HAS_DIAGNOSIS",
        "seqNum":    df["seq_num"].astype(int)
    })
    save(out, os.path.join(RELS_DIR, "has_diagnosis.csv"))

def build_has_lab_result():
    labs = pd.read_csv(os.path.join(CLEANED_DIR, "labevents.csv"), low_memory=False)
    # hadm_id == -1 Indicates unrelated hospitalisation
    labs = labs[labs["hadm_id"] != -1]
    out = pd.DataFrame({
        ":START_ID":    "a_" + labs["hadm_id"].astype(int).astype(str),
        ":END_ID":      "lt_" + labs["itemid"].astype(str),
        ":TYPE":        "HAS_LAB_RESULT",
        "value":        labs["valuenum"].fillna(-999),
        "abnormalFlag": labs["flag"].fillna(""),
        "chartTime":    labs["charttime"].fillna("")
    })
    save(out, os.path.join(RELS_DIR, "has_lab_result.csv"))

def build_derived_from():
    """Generate derived_from.csv from lineage.json(Field → Field)"""
    with open(LINEAGE_PATH, 'r', encoding='utf-8') as f:
        lineage = json.load(f)

    rows = []
    for rec in lineage["lineage_records"]:
        for target in rec["target_fields"]:
            for source in rec["source_fields"]:
                if target != source:   # Exclude imputations where source equals target
                    rows.append({
                        ":START_ID":      target,
                        ":END_ID":        source,
                        ":TYPE":          "DERIVED_FROM",
                        "derivationType": rec["transform_type"]
                    })

    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[":START_ID", ":END_ID", ":TYPE", "derivationType"])
    save(out, os.path.join(RELS_DIR, "derived_from.csv"))

def build_transformed_by():
    """Generate transformed_by.csv from lineage.json(Field → Transformation)"""
    with open(LINEAGE_PATH, 'r', encoding='utf-8') as f:
        lineage = json.load(f)

    rows = []
    for rec in lineage["lineage_records"]:
        for field in rec["target_fields"]:
            rows.append({
                ":START_ID": field,
                ":END_ID":   rec["transformation_id"],
                ":TYPE":     "TRANSFORMED_BY"
            })

    out = pd.DataFrame(rows)
    save(out, os.path.join(RELS_DIR, "transformed_by.csv"))


# ─────────────────────────────────────────────
# Main entrance
# ─────────────────────────────────────────────

if __name__ == "__main__":
    makedirs()

    print("\n=== Nodes ===")
    build_patients()
    build_admissions()
    build_diagnoses()
    build_labtests()
    build_fields()
    build_transformations()

    print("\n=== Relationships ===")
    build_had_admission()
    build_has_diagnosis()
    build_has_lab_result()
    build_derived_from()
    build_transformed_by()

    print("\ngraph_input Completed")