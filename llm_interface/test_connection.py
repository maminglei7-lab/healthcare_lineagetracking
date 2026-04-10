"""
Test Neo4j connection and verify graph data.
Run: python llm_interface/test_connection.py
"""

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from neo4j import GraphDatabase

def test_connection():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # Test 1: Node count by label
        print("=" * 50)
        print("Test 1: Node counts")
        print("=" * 50)
        result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC")
        for r in result:
            print(f"  {r['label']}: {r['count']}")
        
        # Test 2: Relationship count
        print("\n" + "=" * 50)
        print("Test 2: Relationship counts")
        print("=" * 50)
        result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count")
        for r in result:
            print(f"  {r['type']}: {r['count']}")
        
        # Test 3: Sample query - one patient's diagnoses
        print("\n" + "=" * 50)
        print("Test 3: Sample patient diagnoses")
        print("=" * 50)
        result = session.run("""
            MATCH (p:Patient)-[:HAD_ADMISSION]->(a)-[:HAS_DIAGNOSIS]->(d)
            RETURN p.subjectId AS patient, d.icdCode AS icd, d.icdTitle AS diagnosis
            LIMIT 5
        """)
        for r in result:
            print(f"  Patient {r['patient']}: [{r['icd']}] {r['diagnosis']}")
    
    driver.close()
    print("\nNeo4j connection successful!")

if __name__ == "__main__":
    test_connection()