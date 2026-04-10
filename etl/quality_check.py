"""
Quality Check: Validate cleaned data against EDA-driven rules.
Runs after transform, before build_graph_input.
Rules mirror the cleaning logic — if transform did it right, all checks pass.

Two checking modes:
  - Small files : full load into memory
  - Large files : chunked reading, statistics accumulated across chunks
  Large files are defined in config.CHUNKED_FILES.
"""

import pandas as pd
import os
from config import CLEANED_DIR, CHUNKED_FILES, CHUNK_SIZE


class QualityReport:
    def __init__(self, table_name):
        self.table_name = table_name
        self.results = []

    def check(self, rule_name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "table":  self.table_name,
            "rule":   rule_name,
            "status": status,
            "detail": detail,
        })

    def summary(self):
        total  = len(self.results)
        failed = [r for r in self.results if r["status"] == "FAIL"]
        print(f"\n[Quality] {self.table_name}: {total - len(failed)}/{total} passed")
        for r in self.results:
            detail = f" — {r['detail']}" if r["detail"] else ""
            print(f"  {r['status']}  {r['rule']}{detail}")
        return len(failed) == 0


# ─────────────────────────────────────────────
# Small file checkers (full in-memory load)
# ─────────────────────────────────────────────

def check_patients(df):
    r = QualityReport("patients")
    r.check("subject_id is unique",
            df["subject_id"].is_unique)
    r.check("subject_id has no nulls",
            df["subject_id"].notna().all())
    r.check("gender only contains M/F",
            df["gender"].isin(["M", "F"]).all(),
            f"values: {df['gender'].unique().tolist()}")
    r.check("anchor_age in range 0-120",
            df["anchor_age"].between(0, 120).all(),
            f"min={df['anchor_age'].min()}, max={df['anchor_age'].max()}")
    r.check("dod nulls preserved (null = alive)",
            df["dod"].isna().any(),
            f"null count: {df['dod'].isna().sum()}")
    return r.summary()


def check_admissions(df):
    r = QualityReport("admissions")
    r.check("hadm_id is unique",
            df["hadm_id"].is_unique)
    r.check("subject_id has no nulls",
            df["subject_id"].notna().all())
    r.check("admittime has no nulls",
            df["admittime"].notna().all())
    r.check("dischtime has no nulls",
            df["dischtime"].notna().all())
    r.check("language has no '?' values",
            (df["language"] == "?").sum() == 0,
            f"'?' count: {(df['language'] == '?').sum()}")
    r.check("marital_status has no nulls (filled with UNKNOWN)",
            df["marital_status"].notna().all())
    r.check("discharge_location has no nulls (filled with UNKNOWN)",
            df["discharge_location"].notna().all())
    r.check("hospital_expire_flag only contains 0/1",
            df["hospital_expire_flag"].isin([0, 1]).all(),
            f"values: {df['hospital_expire_flag'].unique().tolist()}")
    r.check("deathtime nulls preserved (not filled)",
            df["deathtime"].isna().any(),
            f"null count: {df['deathtime'].isna().sum()}")
    return r.summary()


def check_diagnoses(df):
    r = QualityReport("diagnoses_icd")
    r.check("diagnosis_id column exists",
            "diagnosis_id" in df.columns)
    r.check("icd_version only contains 9 or 10",
            df["icd_version"].isin([9, 10]).all(),
            f"values: {df['icd_version'].unique().tolist()}")
    r.check("icd_code has no nulls",
            df["icd_code"].notna().all())
    r.check("icd_code has no leading/trailing spaces",
            (df["icd_code"] == df["icd_code"].str.strip()).all())
    r.check("diagnosis_id has correct format (_v9 or _v10)",
            df["diagnosis_id"].str.contains(r"_v(?:9|10)$", regex=True).all())
    r.check("no duplicates on (subject_id, hadm_id, seq_num)",
            not df.duplicated(subset=["subject_id", "hadm_id", "seq_num"]).any())
    return r.summary()


def check_d_icd_diagnoses(df):
    r = QualityReport("d_icd_diagnoses")
    r.check("icd_code has no nulls",
            df["icd_code"].notna().all())
    r.check("icd_code has no leading/trailing spaces",
            (df["icd_code"] == df["icd_code"].str.strip()).all())
    r.check("long_title has no nulls",
            df["long_title"].notna().all())
    r.check("icd_version only contains 9 or 10",
            df["icd_version"].isin([9, 10]).all())
    return r.summary()


def check_d_labitems(df):
    r = QualityReport("d_labitems")
    r.check("itemid is unique",
            df["itemid"].is_unique)
    r.check("label has no nulls (filled with Unknown)",
            df["label"].notna().all())
    r.check("label has no empty strings (cleaned)",
            (df["label"].str.strip() != "").all(),
            f"empty count: {(df['label'].str.strip() == '').sum()}")
    r.check("category has no nulls",
            df["category"].notna().all())
    return r.summary()


# ─────────────────────────────────────────────
# Large file checkers (chunked, streaming)
# ─────────────────────────────────────────────

def check_labevents_chunked(path):
    """
    Accumulate statistics across chunks, then evaluate all rules at end.
    Uniqueness of labevent_id is checked within each chunk only — cross-chunk
    uniqueness is assumed from source data guarantee.
    """
    r = QualityReport("labevents")

    total_rows          = 0
    hadm_nulls          = 0
    hadm_non_int        = 0
    flag_nulls          = 0
    flag_unexpected     = 0
    itemid_nulls        = 0
    charttime_nulls     = 0
    valuenum_nulls      = 0
    valueuom_empty      = 0
    ref_both_present    = 0
    outlier_col_missing = False
    intra_chunk_dup     = 0
    first_chunk         = True

    for chunk in pd.read_csv(path, low_memory=False, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)

        if first_chunk:
            if "valuenum_outlier" not in chunk.columns:
                outlier_col_missing = True
            first_chunk = False

        hadm_nulls      += chunk["hadm_id"].isna().sum()
        hadm_non_int    += (~chunk["hadm_id"].apply(
                             lambda x: float(x).is_integer()
                             if pd.notna(x) else True)).sum()
        flag_nulls      += chunk["flag"].isna().sum()
        flag_unexpected += (~chunk["flag"].isin(["normal", "abnormal"])).sum()
        itemid_nulls    += chunk["itemid"].isna().sum()
        charttime_nulls += chunk["charttime"].isna().sum()
        valuenum_nulls  += chunk["valuenum"].isna().sum()
        valueuom_empty  += (chunk["valueuom"].astype(str).str.strip() == "").sum()
        ref_both_present += (
            chunk["ref_range_lower"].notna() & chunk["ref_range_upper"].notna()
        ).sum()
        intra_chunk_dup += chunk.duplicated(subset=["labevent_id"]).sum()

    ref_pct = ref_both_present / total_rows * 100 if total_rows else 0

    r.check("labevent_id unique within each chunk (no intra-chunk dups)",
            intra_chunk_dup == 0,
            f"intra-chunk duplicates: {intra_chunk_dup:,}")
    r.check("hadm_id has no nulls (null rows filtered out)",
            hadm_nulls == 0,
            f"null count: {hadm_nulls:,}")
    r.check("hadm_id is integer type",
            hadm_non_int == 0,
            f"non-integer count: {hadm_non_int:,}")
    r.check("flag has no nulls (filled with 'normal')",
            flag_nulls == 0,
            f"null count: {flag_nulls:,}")
    r.check("flag only contains 'normal' or 'abnormal'",
            flag_unexpected == 0,
            f"unexpected count: {flag_unexpected:,}")
    r.check("itemid has no nulls",
            itemid_nulls == 0,
            f"null count: {itemid_nulls:,}")
    r.check("charttime has no nulls",
            charttime_nulls == 0,
            f"null count: {charttime_nulls:,}")
    r.check("valuenum nulls preserved (not imputed)",
            valuenum_nulls > 0,
            f"null count: {valuenum_nulls:,}")
    r.check("valueuom has no empty strings (cleaned)",
            valueuom_empty == 0,
            f"empty string count: {valueuom_empty:,}")
    r.check("valuenum_outlier column exists",
            not outlier_col_missing)
    r.check("ref_range coverage is reasonable (>70%)",
            ref_pct > 70,
            f"both present: {ref_both_present:,}/{total_rows:,} ({ref_pct:.1f}%)")

    return r.summary()


def check_prescriptions_chunked(path):
    r = QualityReport("prescriptions")

    total_rows      = 0
    hadm_nulls      = 0
    drug_nulls      = 0
    drug_spaces     = 0
    route_nulls     = 0
    drug_type_bad   = 0
    starttime_nulls = 0

    for chunk in pd.read_csv(path, low_memory=False, chunksize=CHUNK_SIZE):
        total_rows      += len(chunk)
        hadm_nulls      += chunk["hadm_id"].isna().sum()
        drug_nulls      += chunk["drug"].isna().sum()
        drug_spaces     += (chunk["drug"] != chunk["drug"].str.strip()).sum()
        route_nulls     += chunk["route"].isna().sum()
        drug_type_bad   += (~chunk["drug_type"].isin(["MAIN", "BASE", "ADDITIVE"])).sum()
        starttime_nulls += chunk["starttime"].isna().sum()

    r.check("hadm_id has no nulls",
            hadm_nulls == 0,
            f"null count: {hadm_nulls:,}")
    r.check("drug has no nulls",
            drug_nulls == 0,
            f"null count: {drug_nulls:,}")
    r.check("drug has no leading/trailing spaces",
            drug_spaces == 0,
            f"count with spaces: {drug_spaces:,}")
    r.check("route has no nulls (filled with UNKNOWN)",
            route_nulls == 0,
            f"null count: {route_nulls:,}")
    r.check("drug_type only contains expected values",
            drug_type_bad == 0,
            f"unexpected count: {drug_type_bad:,}")
    r.check("starttime null rate is acceptable (<1%)",
            starttime_nulls / total_rows < 0.01,
            f"null count: {starttime_nulls:,} ({starttime_nulls/total_rows*100:.2f}%)")

    return r.summary()


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

SMALL_CHECKERS = {
    "patients":        check_patients,
    "admissions":      check_admissions,
    "diagnoses_icd":   check_diagnoses,
    "d_icd_diagnoses": check_d_icd_diagnoses,
    "d_labitems":      check_d_labitems,
}

CHUNKED_CHECKERS = {
    "labevents":     check_labevents_chunked,
    "prescriptions": check_prescriptions_chunked,
}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_quality_checks(cleaned_dir=CLEANED_DIR):
    all_passed = True

    # Small files — full load
    for name, checker in SMALL_CHECKERS.items():
        path = os.path.join(cleaned_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"\n[Quality] {name}: SKIPPED — file not found")
            continue
        df = pd.read_csv(path, low_memory=False)
        if not checker(df):
            all_passed = False

    # Large files — chunked
    for name, checker in CHUNKED_CHECKERS.items():
        path = os.path.join(cleaned_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"\n[Quality] {name}: SKIPPED — file not found")
            continue
        if not checker(path):
            all_passed = False

    print("\n" + "=" * 50)
    print("  All quality checks PASSED" if all_passed else "  Some checks FAILED")
    print("=" * 50)
    return all_passed


if __name__ == "__main__":
    run_quality_checks()