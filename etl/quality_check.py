import pandas as pd
import os

CLEANED_DIR = r"D:\Desktop\project\data\cleaned"

class QualityReport:
    def __init__(self, table_name):
        self.table_name = table_name
        self.results = []

    def check(self, rule_name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "table": self.table_name,
            "rule": rule_name,
            "status": status,
            "detail": detail
        })

    def summary(self):
        total = len(self.results)
        failed = [r for r in self.results if "FAIL" in r["status"]]
        print(f"\n[Quality] {self.table_name}: {total - len(failed)}/{total} passed")
        for r in self.results:
            print(f"  {r['status']}  {r['rule']}" + (f" — {r['detail']}" if r["detail"] else ""))
        return len(failed) == 0


# ─────────────────────────────────────────────
# Quality Rules
# ─────────────────────────────────────────────

def check_patients(df):
    r = QualityReport("patients")
    r.check("subject_id should be unique",
            df["subject_id"].is_unique)
    r.check("anchor_age range between 0-120",
            df["anchor_age"].between(0, 120).all(),
            f"min={df['anchor_age'].min()}, max={df['anchor_age'].max()}")
    r.check("gender only include M/F",
            df["gender"].isin(["M", "F"]).all(),
            f"unique values: {df['gender'].unique().tolist()}")
    r.check("No null values in dod field(filled with unknown)",
            df["dod"].notna().all())
    return r.summary()

def check_admissions(df):
    r = QualityReport("admissions")
    r.check("hadm_id should be unique",
            df["hadm_id"].is_unique)
    r.check("subject_id should not have null value",
            df["subject_id"].notna().all())
    r.check("language should not includ '?'",
            (df["language"] == "?").sum() == 0,
            f"'?' count: {(df['language'] == '?').sum()}")
    r.check("marital_status should not have null value(filled with unknown)",
            df["marital_status"].notna().all())
    r.check("hospital_expire_flag only include 0/1",
            df["hospital_expire_flag"].isin([0, 1]).all(),
            f"unique values: {df['hospital_expire_flag'].unique().tolist()}")
    return r.summary()

def check_diagnoses(df):
    r = QualityReport("diagnoses_icd")
    r.check("diagnosis_id exist",
            "diagnosis_id" in df.columns)
    r.check("icd_version only include 9 or 10",
            df["icd_version"].isin([9, 10]).all(),
            f"unique values: {df['icd_version'].unique().tolist()}")
    r.check("icd_code should not have null value",
            df["icd_code"].notna().all())
    r.check("diagnosis_id has right format( _v9 or _v10)",
            df["diagnosis_id"].str.contains(r"_v(9|10)$", regex=True).all())
    return r.summary()

def check_labevents(df):
    r = QualityReport("labevents")
    r.check("labevent_id should not be unique",
            df["labevent_id"].is_unique)
    r.check("valuenum should not have null value(filled with median)",
            df["valuenum"].notna().all())
    r.check("flag should not have null value(filled with normal)",
            df["flag"].notna().all())
    r.check("hadm_id should not have null value(filled with -1)",
            df["hadm_id"].notna().all())
    r.check("itemid should not have null value",
            df["itemid"].notna().all())
    return r.summary()

def check_d_icd_diagnoses(df):
    r = QualityReport("d_icd_diagnoses")
    r.check("icd_code should not have null value", df["icd_code"].notna().all())
    r.check("long_title should not have null value", df["long_title"].notna().all())
    return r.summary()

def check_d_labitems(df):
    r = QualityReport("d_labitems")
    r.check("itemid should be unique", df["itemid"].is_unique)
    r.check("label should not have null value", df["label"].notna().all())
    return r.summary()


# ─────────────────────────────────────────────
# Main Entrance
# ─────────────────────────────────────────────

def run_quality_checks():
    tables = {
        "patients":        "patients.csv",
        "admissions":      "admissions.csv",
        "diagnoses_icd":   "diagnoses_icd.csv",
        "labevents":       "labevents.csv",
        "d_icd_diagnoses": "d_icd_diagnoses.csv",
        "d_labitems":      "d_labitems.csv",
    }

    checkers = {
        "patients":        check_patients,
        "admissions":      check_admissions,
        "diagnoses_icd":   check_diagnoses,
        "labevents":       check_labevents,
        "d_icd_diagnoses": check_d_icd_diagnoses,
        "d_labitems":      check_d_labitems,
    }

    all_passed = True
    for name, fname in tables.items():
        path = os.path.join(CLEANED_DIR, fname)
        df = pd.read_csv(path, low_memory=False)
        passed = checkers[name](df)
        if not passed:
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("All quality checks pass")
    else:
        print("Partial checks failed, check the FAIL items above.")
    print("=" * 40)

if __name__ == "__main__":
    run_quality_checks()