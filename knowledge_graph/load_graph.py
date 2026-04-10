"""
Load Graph: Import nodes and relationships from CSV into Neo4j.

Schema:
  Layer 1 — Clinical Entities:
    (Patient)-[HAD_ADMISSION]->(Admission)
    (Admission)-[HAS_DIAGNOSIS]->(Diagnosis)
    (Admission)-[HAS_LAB_RESULT]->(LabTest)
    (Admission)-[HAS_PRESCRIPTION]->(Medication)

  Layer 2 — Knowledge Enhancement:
    (Diagnosis)-[BELONGS_TO_CATEGORY]->(ICD_Category)

Large relationship files (has_lab_result 84.6M, has_prescription 20.3M)
are read in chunks to avoid OOM. CHUNK_SIZE is read from config.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "etl"))

import pandas as pd
from neo4j import GraphDatabase
from config import (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
                    NODES_DIR, RELS_DIR, CHUNK_SIZE)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

BATCH_SIZE = 5_000   # rows per Neo4j UNWIND transaction


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def run_batch(session, query, rows):
    """Write rows to Neo4j in batches of BATCH_SIZE."""
    for i in range(0, len(rows), BATCH_SIZE):
        session.run(query, rows=rows[i:i + BATCH_SIZE])


def load_csv(dir_path, filename):
    """Full in-memory load for small files."""
    return pd.read_csv(f"{dir_path}/{filename}", low_memory=False)


def load_csv_chunked(dir_path, filename):
    """Chunked generator for large files."""
    return pd.read_csv(
        f"{dir_path}/{filename}", low_memory=False, chunksize=CHUNK_SIZE
    )


def to_records(df, rename_map, drop_cols=None):
    """Rename ID columns and convert to list of dicts."""
    df = df.rename(columns=rename_map)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df.where(df.notna(), None).to_dict("records")


# ─────────────────────────────────────────────
# 1. Clear & Setup
# ─────────────────────────────────────────────

def clear_db(session):
    print("Clearing database in batches...")
    while True:
        result = session.run(
            "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS deleted"
        )
        deleted = result.single()["deleted"]
        print(f"  Deleted {deleted:,} nodes...")
        if deleted == 0:
            break
    print("✓ Database cleared")


def create_constraints(session):
    constraints = [
        ("Patient",      "patientId"),
        ("Admission",    "admissionId"),
        ("Diagnosis",    "diagnosisId"),
        ("LabTest",      "labTestId"),
        ("Medication",   "medicationId"),
        ("ICD_Category", "categoryId"),
    ]
    for label, prop in constraints:
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )
    print(f"✓ {len(constraints)} constraints created")


# ─────────────────────────────────────────────
# 2. Layer 1 Nodes  (all small files — full load)
# ─────────────────────────────────────────────

def load_patients(session):
    rows = to_records(load_csv(NODES_DIR, "patients.csv"),
                      {"patientId:ID": "patientId"}, [":LABEL"])
    run_batch(session,
        "UNWIND $rows AS r "
        "MERGE (p:Patient {patientId: r.patientId}) "
        "SET p.subjectId = r.subjectId, p.gender = r.gender, "
        "    p.anchorAge = r.anchorAge, p.anchorYear = r.anchorYear",
        rows)
    print(f"✓ Patients: {len(rows):,}")


def load_admissions(session):
    rows = to_records(load_csv(NODES_DIR, "admissions.csv"),
                      {"admissionId:ID": "admissionId"}, [":LABEL"])
    run_batch(session,
        "UNWIND $rows AS r "
        "MERGE (a:Admission {admissionId: r.admissionId}) "
        "SET a.hadmId = r.hadmId, a.admissionType = r.admissionType, "
        "    a.admitTime = r.admitTime, a.dischargeTime = r.dischargeTime, "
        "    a.insurance = r.insurance, a.language = r.language, "
        "    a.maritalStatus = r.maritalStatus, a.race = r.race",
        rows)
    print(f"✓ Admissions: {len(rows):,}")


def load_diagnoses(session):
    rows = to_records(load_csv(NODES_DIR, "diagnoses.csv"),
                      {"diagnosisId:ID": "diagnosisId"}, [":LABEL"])
    run_batch(session,
        "UNWIND $rows AS r "
        "MERGE (d:Diagnosis {diagnosisId: r.diagnosisId}) "
        "SET d.icdCode = r.icdCode, d.icdVersion = r.icdVersion, "
        "    d.icdTitle = r.icdTitle",
        rows)
    print(f"✓ Diagnoses: {len(rows):,}")


def load_labtests(session):
    rows = to_records(load_csv(NODES_DIR, "labtests.csv"),
                      {"labTestId:ID": "labTestId"}, [":LABEL"])
    run_batch(session,
        "UNWIND $rows AS r "
        "MERGE (l:LabTest {labTestId: r.labTestId}) "
        "SET l.itemId = r.itemId, l.label = r.label, "
        "    l.fluid = r.fluid, l.category = r.category",
        rows)
    print(f"✓ LabTests: {len(rows):,}")


def load_medications(session):
    rows = to_records(load_csv(NODES_DIR, "medications.csv"),
                      {"medicationId:ID": "medicationId"}, [":LABEL"])
    run_batch(session,
        "UNWIND $rows AS r "
        "MERGE (m:Medication {medicationId: r.medicationId}) "
        "SET m.drugName = r.drugName",
        rows)
    print(f"✓ Medications: {len(rows):,}")


# ─────────────────────────────────────────────
# 3. Layer 2 Nodes
# ─────────────────────────────────────────────

def load_icd_categories(session):
    rows = to_records(load_csv(NODES_DIR, "icd_categories.csv"),
                      {"categoryId:ID": "categoryId"}, [":LABEL"])
    run_batch(session,
        "UNWIND $rows AS r "
        "MERGE (c:ICD_Category {categoryId: r.categoryId}) "
        "SET c.code = r.code, c.icdVersion = r.icdVersion, c.title = r.title",
        rows)
    print(f"✓ ICD_Categories: {len(rows):,}")


# ─────────────────────────────────────────────
# 4. Layer 1 Relationships
# ─────────────────────────────────────────────

def load_had_admission(session):
    rows = to_records(load_csv(RELS_DIR, "had_admission.csv"),
                      {":START_ID": "start", ":END_ID": "end"}, [":TYPE"])
    run_batch(session,
        "UNWIND $rows AS r "
        "MATCH (p:Patient {patientId: r.start}), (a:Admission {admissionId: r.end}) "
        "MERGE (p)-[:HAD_ADMISSION]->(a)",
        rows)
    print(f"✓ HAD_ADMISSION: {len(rows):,}")


def load_has_diagnosis(session):
    rows = to_records(load_csv(RELS_DIR, "has_diagnosis.csv"),
                      {":START_ID": "start", ":END_ID": "end"}, [":TYPE"])
    run_batch(session,
        "UNWIND $rows AS r "
        "MATCH (a:Admission {admissionId: r.start}), (d:Diagnosis {diagnosisId: r.end}) "
        "MERGE (a)-[rel:HAS_DIAGNOSIS]->(d) "
        "SET rel.seqNum = r.seqNum",
        rows)
    print(f"✓ HAS_DIAGNOSIS: {len(rows):,}")


def load_has_lab_result(session):
    """84.6M rows — chunked read + batched Neo4j writes."""
    total = 0
    for chunk in load_csv_chunked(RELS_DIR, "has_lab_result.csv"):
        rows = to_records(chunk, {":START_ID": "start", ":END_ID": "end"}, [":TYPE"])
        run_batch(session,
            "UNWIND $rows AS r "
            "MATCH (a:Admission {admissionId: r.start}), (l:LabTest {labTestId: r.end}) "
            "CREATE (a)-[rel:HAS_LAB_RESULT]->(l) "
            "SET rel.value = r.value, rel.flag = r.flag, "
            "    rel.chartTime = r.chartTime, "
            "    rel.refRangeLower = r.refRangeLower, "
            "    rel.refRangeUpper = r.refRangeUpper, "
            "    rel.valueuom = r.valueuom, "
            "    rel.outlier = r.outlier",
            rows)
        total += len(rows)
        print(f"  HAS_LAB_RESULT: {total:,} loaded...", end="\r")
    print(f"\n✓ HAS_LAB_RESULT: {total:,}")


def load_has_prescription(session):
    """20.3M rows — chunked read + batched Neo4j writes."""
    total = 0
    for chunk in load_csv_chunked(RELS_DIR, "has_prescription.csv"):
        rows = to_records(chunk, {":START_ID": "start", ":END_ID": "end"}, [":TYPE"])
        run_batch(session,
            "UNWIND $rows AS r "
            "MATCH (a:Admission {admissionId: r.start}), (m:Medication {medicationId: r.end}) "
            "CREATE (a)-[rel:HAS_PRESCRIPTION]->(m) "
            "SET rel.drugType = r.drugType, rel.doseVal = r.doseVal, "
            "    rel.doseUnit = r.doseUnit, rel.route = r.route, "
            "    rel.startTime = r.startTime, rel.stopTime = r.stopTime",
            rows)
        total += len(rows)
        print(f"  HAS_PRESCRIPTION: {total:,} loaded...", end="\r")
    print(f"\n✓ HAS_PRESCRIPTION: {total:,}")


# ─────────────────────────────────────────────
# 5. Layer 2 Relationships
# ─────────────────────────────────────────────

def load_belongs_to_category(session):
    rows = to_records(load_csv(RELS_DIR, "belongs_to_category.csv"),
                      {":START_ID": "start", ":END_ID": "end"}, [":TYPE"])
    run_batch(session,
        "UNWIND $rows AS r "
        "MATCH (d:Diagnosis {diagnosisId: r.start}), (c:ICD_Category {categoryId: r.end}) "
        "MERGE (d)-[:BELONGS_TO_CATEGORY]->(c)",
        rows)
    print(f"✓ BELONGS_TO_CATEGORY: {len(rows):,}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    with driver.session() as session:

        print("=== Clear Database ===")
        clear_db(session)

        print("\n=== Create Constraints ===")
        create_constraints(session)

        print("\n=== Load Layer 1 Nodes ===")
        load_patients(session)
        load_admissions(session)
        load_diagnoses(session)
        load_labtests(session)
        load_medications(session)

        print("\n=== Load Layer 2 Nodes ===")
        load_icd_categories(session)

        print("\n=== Load Layer 1 Relationships ===")
        load_had_admission(session)
        load_has_diagnosis(session)
        load_has_lab_result(session)
        load_has_prescription(session)

        print("\n=== Load Layer 2 Relationships ===")
        load_belongs_to_category(session)

        print("\n=== Graph Summary ===")
        total_nodes = 0
        for r in session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"
        ):
            print(f"  {r['label']}: {r['cnt']:,}")
            total_nodes += r["cnt"]

        total_rels = 0
        for r in session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS cnt ORDER BY cnt DESC"
        ):
            print(f"  {r['type']}: {r['cnt']:,}")
            total_rels += r["cnt"]

        print(f"\n  Total: {total_nodes:,} nodes, {total_rels:,} relationships")
        print("Graph loaded successfully")

    driver.close()