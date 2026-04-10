"""
Check Neo4j graph schema: node properties and relationship properties.
Run: python check_schema.py
"""

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from neo4j import GraphDatabase

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
session = driver.session()

print("=== Node Properties ===")
for label in ["Patient", "Admission", "Diagnosis", "Medication", "LabTest", "ICD_Category"]:
    result = session.run(f"MATCH (n:{label}) RETURN n LIMIT 1")
    rec = result.single()
    if rec:
        props = dict(rec["n"])
        print(f"\n{label}:")
        for k, v in props.items():
            print(f"  {k}: {v}")

print("\n=== Relationship Properties ===")
for rel in ["HAD_ADMISSION", "HAS_DIAGNOSIS", "HAS_LAB_RESULT", "HAS_PRESCRIPTION", "BELONGS_TO_CATEGORY"]:
    result = session.run(f"MATCH ()-[r:{rel}]->() RETURN r LIMIT 1")
    rec = result.single()
    if rec:
        props = dict(rec["r"])
        print(f"\n{rel}:")
        if props:
            for k, v in props.items():
                print(f"  {k}: {v}")
        else:
            print("  (no properties)")

driver.close()