"""
Phase 4: Demo Scenario Testing
Covers all 5 query types from the project proposal.
Run: python test_scenarios.py
"""

from graph_rag import GraphRAGPipeline
import json
import time

def run_test(pipeline, scenario_name, query_type, question):
    """Run a single test scenario and print results."""
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"Type: {query_type}")
    print(f"Question: {question}")
    print(f"{'='*70}")

    start = time.time()
    result = pipeline.query(question)
    elapsed = time.time() - start

    print(f"\n⏱️ Response time: {elapsed:.2f}s")
    print(f"📦 Records returned: {len(result['results'])}")

    # Check if query succeeded
    has_error = any("error" in str(r).lower() for r in result["results"])
    if has_error or len(result["results"]) == 0:
        print("❌ FAILED - No results or error")
    else:
        print("✅ PASSED")

    return {
        "scenario": scenario_name,
        "type": query_type,
        "question": question,
        "cypher": result["cypher"],
        "record_count": len(result["results"]),
        "time": round(elapsed, 2),
        "passed": not has_error and len(result["results"]) > 0,
        "answer": result["answer"],
    }


def main():
    pipeline = GraphRAGPipeline()
    results = []

    # =========================================================
    # Scenario 1: Single-hop Factual Query
    # Pattern: Patient → Admission → Diagnosis
    # =========================================================
    results.append(run_test(
        pipeline,
        scenario_name="1. Single-hop Factual",
        query_type="Patient → Admission → Diagnosis",
        question="What diagnoses does patient 10014729 have?"
    ))

    # =========================================================
    # Scenario 2: Multi-hop Relational Analysis
    # Pattern: Diagnosis → Admission → Medication
    # =========================================================
    results.append(run_test(
        pipeline,
        scenario_name="2. Multi-hop Relational",
        query_type="ICD_Category(angina) → Diagnosis → Admission → Medication",
        question="What medications were prescribed for patients with angina?"
    ))

    # =========================================================
    # Scenario 3: Aggregation with Condition
    # Pattern: Patient → Admission → LabTest (filter abnormal)
    # =========================================================
    results.append(run_test(
        pipeline,
        scenario_name="3. Aggregation + Condition",
        query_type="Patient → Admission → LabTest (flag=abnormal) → COUNT",
        question="How many abnormal lab results does patient 10014729 have?"
    ))

    # =========================================================
    # Scenario 4: Backward Tracing (Reverse Query)
    # Pattern: Medication → Admission → Patient → Aggregate
    # =========================================================
    results.append(run_test(
        pipeline,
        scenario_name="4. Backward Tracing",
        query_type="Medication(Aspirin) → Admission → Patient → Aggregate",
        question="What are the common characteristics of patients who were prescribed Aspirin? Include their average age, gender distribution, and most common diagnoses."
    ))

    # =========================================================
    # Scenario 5: Complex Comprehensive Query
    # Pattern: ICD_Category → Diagnosis → Admission → LabTest → Aggregate
    # (This is the "Complete Example" from the project proposal)
    # =========================================================
    results.append(run_test(
        pipeline,
        scenario_name="5. Complex Comprehensive",
        query_type="ICD_Category(heart failure) → Diagnosis → Admission → LabTest(abnormal) → Aggregate",
        question="What are the most common abnormal lab indicators for heart failure patients?"
    ))

    # =========================================================
    # Bonus Scenario 6: Cross-category Aggregation
    # Pattern: ICD_Category → Diagnosis → Admission → Patient → GROUP BY
    # =========================================================
    results.append(run_test(
        pipeline,
        scenario_name="6. Cross-category Aggregation",
        query_type="ICD_Category → Diagnosis → Admission → Patient → GROUP BY age",
        question="What is the age distribution of patients by their top diagnosis categories?"
    ))

    # =========================================================
    # Summary Report
    # =========================================================
    print("\n" + "=" * 70)
    print("📋 PHASE 4 TEST SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<30} {'Records':>8} {'Time':>8} {'Status':>8}")
    print("-" * 70)
    passed = 0
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['scenario']:<30} {r['record_count']:>8} {r['time']:>7.2f}s {status:>8}")
        if r["passed"]:
            passed += 1
    print("-" * 70)
    print(f"Total: {passed}/{len(results)} passed")
    print("=" * 70)

    # Save detailed results
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n📄 Detailed results saved to test_results.json")

    pipeline.close()


if __name__ == "__main__":
    main()