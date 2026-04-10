"""
Graph RAG Pipeline - Core Module
Three-stage pipeline with query decomposition:
  Stage 1a: Query Decomposition (split compound questions)
  Stage 1b: Cypher Generation (per sub-question)
  Stage 2:  Graph Retrieval (Neo4j)
  Stage 3:  Evidence-based Answer Generation (LLM)

Run: python graph_rag.py
"""

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY, get_llm_config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase
import json
import re

# =============================================================
# Graph Schema Description
# =============================================================
GRAPH_SCHEMA = """
Neo4j Graph Schema:

Node Labels and Properties:
- Patient: patientId (string, format "p_10014729"), subjectId (integer, format 10014729), gender, anchorAge, anchorYear
- Admission: hadmId (integer, use this to match specific admissions e.g. {hadmId: 26415640}), admissionId (internal ID, do NOT use for matching), admitTime, dischargeTime, admissionType, insurance, language, maritalStatus, race
- Diagnosis: diagnosisId, icdCode, icdVersion, icdTitle
- Medication: medicationId, drugName
- LabTest: labTestId, itemId, label, fluid, category
- ICD_Category: categoryId, code, icdVersion, title

Relationships:
- (Patient)-[:HAD_ADMISSION]->(Admission)
- (Admission)-[:HAS_DIAGNOSIS {seqNum}]->(Diagnosis)
- (Admission)-[:HAS_LAB_RESULT {value, flag, outlier, refRangeLower, refRangeUpper, chartTime, valueuom}]->(LabTest)
- (Admission)-[:HAS_PRESCRIPTION {doseVal, doseUnit, route, drugType, startTime, stopTime}]->(Medication)
- (Diagnosis)-[:BELONGS_TO_CATEGORY]->(ICD_Category)

Important Notes:
- Patient.patientId format: "p_10014729" (string with p_ prefix)
- Patient.subjectId format: 10014729 (integer, no prefix)
- flag values in HAS_LAB_RESULT: "normal" or "abnormal" (original MIMIC-IV field)
- outlier in HAS_LAB_RESULT: boolean (true = ETL-flagged as statistical outlier based on refRangeLower/Upper)
- IMPORTANT: To filter abnormal/outlier lab results, use r.outlier = true (NOT r.flag = 'abnormal')
- IMPORTANT: To match a specific admission by ID, always use hadmId (integer): {hadmId: 26415640}. Never use admissionId for matching.
- When user mentions a patient number like "10014729", use subjectId (integer) to match
- When aggregating medications, use Medication.drugName
- ICD_Category links Diagnosis to broader disease categories
"""

# =============================================================
# Stage 1a: Query Decomposition
# =============================================================
DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical query analyst. Your job is to determine if a user's question is a compound question that should be split into separate sub-queries for a graph database.

A compound question asks about MULTIPLE DIFFERENT aspects that require DIFFERENT graph traversal paths.
Examples of compound questions:
- "What medications were prescribed for patient X and what were their diagnoses?" → 2 sub-questions (medications path vs diagnoses path)
- "What are the diagnoses and abnormal lab results for patient X?" → 2 sub-questions
- "Compare the medications and lab results for heart failure patients" → 2 sub-questions

Examples of NON-compound questions (single query is enough):
- "What medications were prescribed for heart failure patients?" → single path: Diagnosis→Admission→Medication
- "How many abnormal lab results does patient X have?" → single path with aggregation
- "What are the most common diagnoses?" → single aggregation

Rules:
1. If the question is compound, output each sub-question on a separate line, prefixed with "SUB: ".
2. If the question is NOT compound, output exactly: "SINGLE"
3. Each sub-question must be a complete, self-contained question.
4. Keep shared context (like patient ID) in each sub-question.
5. Maximum 3 sub-questions.
6. Output ONLY the sub-questions or "SINGLE", nothing else.
"""),
    ("human", "{question}")
])

# =============================================================
# Stage 1b: Cypher Generation
# =============================================================
CYPHER_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical knowledge graph expert. Given a user's natural language question, 
generate a Cypher query to retrieve the answer from a Neo4j graph database.

{schema}

Rules:
1. Return ONLY the Cypher query, no explanation, no markdown code blocks.
2. Always use LIMIT to avoid returning too many results (default LIMIT 20).
3. For patient lookups by number, use subjectId (integer): MATCH (p:Patient {{subjectId: 10014729}})
4. For aggregation queries, use COUNT, COLLECT, AVG, etc.
5. When filtering abnormal lab results, use: WHERE r.outlier = true
6. For medication queries, return drugName.
7. Always return meaningful properties, not just node references.
8. IMPORTANT: For disease/diagnosis name matching, NEVER use exact match. Always use CONTAINS or STARTS WITH for fuzzy matching. Example: WHERE d.icdTitle CONTAINS 'angina' (use lowercase with toLower()).
9. When searching by disease category, prefer matching via ICD_Category.title with CONTAINS.
10. IMPORTANT: Always start traversal from the SMALLEST node set first to avoid memory errors.
    HAS_LAB_RESULT has 84M rows and HAS_PRESCRIPTION has 20M rows — never start from these.
    Always filter to a small node set first, then traverse into large relationships last.
    GOOD pattern:
      MATCH (c:ICD_Category) WHERE toLower(c.title) CONTAINS 'heart failure'
      MATCH (d:Diagnosis)-[:BELONGS_TO_CATEGORY]->(c)
      MATCH (a:Admission)-[:HAS_DIAGNOSIS]->(d)
      MATCH (a)-[r:HAS_LAB_RESULT]->(l:LabTest) WHERE r.flag = 'abnormal'
      RETURN l.label, COUNT(*) ORDER BY COUNT(*) DESC LIMIT 20
    BAD pattern:
      MATCH (a)-[r:HAS_LAB_RESULT]->(l), (a)-[:HAS_DIAGNOSIS]->(d)-[:BELONGS_TO_CATEGORY]->(c)
      WHERE r.flag = 'abnormal' AND toLower(c.title) CONTAINS 'heart failure'
11. CRITICAL: To identify abnormal lab results, ALWAYS use r.outlier = true. NEVER use r.flag = 'abnormal'.
12. CRITICAL: When querying medications, ALWAYS filter out Not Formulary entries: WHERE NOT toLower(m.drugName) STARTS WITH '*nf'
13. CRITICAL: To match a specific admission, ALWAYS use hadmId (integer), NEVER use admissionId. Example: MATCH (a:Admission {{hadmId: 26415640}})
"""),
    ("human", "{question}")
])

# =============================================================
# Stage 3: Answer Generation
# =============================================================
ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful medical data assistant. Based on the graph database query results, 
provide a clear, accurate answer to the user's question.

Rules:
1. You MUST answer in the SAME language as the user's question. If the question is in English, answer in English. If in Chinese, answer in Chinese.
2. If the query returned no results, say so clearly and suggest possible reasons.
3. Include specific numbers and data from the results.
4. Briefly describe the reasoning path (which nodes and relationships were traversed).
5. Keep the answer concise but informative.
6. If results come from multiple sub-queries, synthesize them into ONE coherent answer.
"""),
    ("human", """User Question: {question}

{query_details}

Please provide a clear, unified answer based on all the above results.""")
])


class GraphRAGPipeline:
    """Three-stage Graph RAG Pipeline with query decomposition."""

    def __init__(self):
        llm_cfg = get_llm_config()
        self.llm = ChatOpenAI(
            model=llm_cfg["model_name"],
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
            api_key=OPENAI_API_KEY,
        )
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.decompose_chain = DECOMPOSITION_PROMPT | self.llm
        self.cypher_chain = CYPHER_GENERATION_PROMPT | self.llm
        self.answer_chain = ANSWER_GENERATION_PROMPT | self.llm
        print(f"✅ Pipeline initialized (LLM: {llm_cfg['model_name']})")

    # ---------------------------------------------------------
    # Stage 1a: Query Decomposition
    # ---------------------------------------------------------
    def stage1a_decompose(self, question: str) -> list:
        """Decompose compound question into sub-questions. Returns list of questions."""
        print("\n--- Stage 1a: Query Decomposition ---")
        response = self.decompose_chain.invoke({"question": question})
        text = response.content.strip()
        print(f"  Decomposition result: {text}")

        if text.upper() == "SINGLE":
            return [question]

        sub_questions = []
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("SUB:"):
                sub_questions.append(line[4:].strip())
            elif line and not line.upper().startswith("SINGLE"):
                sub_questions.append(line)

        if not sub_questions:
            return [question]

        print(f"  Split into {len(sub_questions)} sub-questions:")
        for i, sq in enumerate(sub_questions):
            print(f"    [{i+1}] {sq}")
        return sub_questions

    # ---------------------------------------------------------
    # Stage 1b: Cypher Generation
    # ---------------------------------------------------------
    def stage1b_generate_cypher(self, question: str) -> str:
        """Generate Cypher for a single question."""
        print(f"\n--- Stage 1b: Cypher Generation ---")
        response = self.cypher_chain.invoke({
            "schema": GRAPH_SCHEMA,
            "question": question,
        })
        cypher = response.content.strip()
        if cypher.startswith("```"):
            cypher = cypher.split("\n", 1)[1]
        if cypher.endswith("```"):
            cypher = cypher.rsplit("```", 1)[0]
        cypher = cypher.strip()
        print(f"  Cypher: {cypher}")
        return cypher

    def stage1_generate_cypher(self, question: str) -> str:
        return self.stage1b_generate_cypher(question)

    # ---------------------------------------------------------
    # Stage 2: Execute Query
    # ---------------------------------------------------------
    def stage2_execute_query(self, cypher: str) -> list:
        """Execute Cypher query against Neo4j."""
        print("\n--- Stage 2: Graph Retrieval ---")
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                records = [dict(r) for r in result]
                print(f"  Retrieved {len(records)} records")
                return records
        except Exception as e:
            print(f"  ❌ Query execution error: {e}")
            return [{"error": str(e)}]

    # ---------------------------------------------------------
    # Stage 2b: Extract Subgraph
    # ---------------------------------------------------------
    def stage2_extract_subgraph(self, cypher: str) -> dict:
        """Extract the subgraph traversed by the query for visualization."""
        print("\n--- Stage 2b: Subgraph Extraction ---")
        subgraph = {"nodes": [], "edges": []}
        seen_nodes = set()
        seen_edges = set()

        try:
            subgraph_cypher = self._build_subgraph_cypher(cypher)
            if not subgraph_cypher:
                return subgraph

            print(f"  Subgraph Cypher: {subgraph_cypher}")
            with self.driver.session() as session:
                result = session.run(subgraph_cypher)
                for record in result:
                    for key in record.keys():
                        val = record[key]
                        if hasattr(val, "labels"):
                            node_id = val.element_id
                            if node_id not in seen_nodes:
                                seen_nodes.add(node_id)
                                label = list(val.labels)[0] if val.labels else "Unknown"
                                props = dict(val)
                                if label == "Patient":
                                    sid = props.get("subjectId", "")
                                    age = props.get("anchorAge", "")
                                    gender = props.get("gender", "")
                                    display = f"Patient {sid}"
                                    if age:
                                        display += f" ({gender}, {age}y)" if gender else f" (age {age})"
                                elif label == "Admission":
                                    admit = props.get("admitTime", "")
                                    atype = props.get("admissionType", "")
                                    date_str = str(admit)[:10] if admit else "unknown"
                                    display = f"Admission {date_str}"
                                    if atype:
                                        display += f" ({atype})"
                                elif label == "Medication":
                                    display = props.get("drugName", str(node_id))
                                elif label == "Diagnosis":
                                    display = props.get("icdTitle", props.get("icdCode", str(node_id)))
                                elif label == "LabTest":
                                    display = props.get("label", props.get("category", str(node_id)))
                                elif label == "ICD_Category":
                                    display = props.get("title", props.get("code", str(node_id)))
                                else:
                                    display = str(node_id)
                                subgraph["nodes"].append({
                                    "id": node_id,
                                    "label": label,
                                    "display": display,
                                    "properties": props,
                                })
                        elif hasattr(val, "type"):
                            edge_id = val.element_id
                            if edge_id not in seen_edges:
                                seen_edges.add(edge_id)
                                subgraph["edges"].append({
                                    "id": edge_id,
                                    "type": val.type,
                                    "source": val.start_node.element_id,
                                    "target": val.end_node.element_id,
                                    "properties": dict(val),
                                })

            print(f"  Extracted {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
        except Exception as e:
            print(f"  ⚠️ Subgraph extraction failed: {e}")

        return subgraph

    def _merge_subgraphs(self, sg1: dict, sg2: dict) -> dict:
        """Merge two subgraphs, deduplicating by id."""
        merged = {"nodes": list(sg1["nodes"]), "edges": list(sg1["edges"])}
        seen_node_ids = set(n["id"] for n in sg1["nodes"])
        seen_edge_ids = set(e["id"] for e in sg1["edges"])
        for n in sg2["nodes"]:
            if n["id"] not in seen_node_ids:
                seen_node_ids.add(n["id"])
                merged["nodes"].append(n)
        for e in sg2["edges"]:
            if e["id"] not in seen_edge_ids:
                seen_edge_ids.add(e["id"])
                merged["edges"].append(e)
        return merged

    def _build_subgraph_cypher(self, original_cypher: str) -> str:
        """Convert original Cypher to a subgraph extraction query.

        Strategy:
        - Keep all MATCH clauses with their immediately following WHERE clauses
        - Drop WITH clauses entirely (not needed for subgraph traversal, and
          they break variable scoping when naively re-ordered)
        - Inject variable names into anonymous relationships so they can be RETURNed
        - Build RETURN from all collected variable names + LIMIT 30
        """
        try:
            lines = [l.strip() for l in original_cypher.split("\n") if l.strip()]
            match_parts = []   # list of (match_line, [where_lines])
            i = 0

            while i < len(lines):
                upper = lines[i].upper()
                if upper.startswith("MATCH"):
                    match_line = lines[i]
                    where_lines = []
                    # Consume any WHERE lines that directly follow
                    j = i + 1
                    while j < len(lines) and lines[j].upper().startswith("WHERE"):
                        where_lines.append(lines[j])
                        j += 1
                    match_parts.append((match_line, where_lines))
                    i = j
                elif upper.startswith(("RETURN", "ORDER", "LIMIT")):
                    break  # stop here
                else:
                    # Skip WITH, standalone WHERE after WITH, etc.
                    i += 1

            if not match_parts:
                return ""

            counter = [0]

            def inject_rel_var(m):
                bracket_content = m.group(1)
                # Already has a variable name (starts with word chars then : or ])
                if re.match(r'^\w+[\s:\]|]', bracket_content):
                    return m.group(0)
                var_name = f"_r{counter[0]}"
                counter[0] += 1
                return f"[{var_name}{bracket_content}]"

            query_parts = []
            all_vars = set()

            for match_line, where_lines in match_parts:
                new_match = re.sub(r'\[([^\]]*)\]', inject_rel_var, match_line)
                query_parts.append(new_match)
                for wl in where_lines:
                    query_parts.append(wl)

                # Collect variable names from this MATCH line
                node_vars = re.findall(r'\((\w+?)(?::|[\s\)])', new_match)
                all_vars.update(v for v in node_vars if v)
                rel_vars = re.findall(r'\[(\w+?)(?::|[\s\]])', new_match)
                all_vars.update(v for v in rel_vars if v)

            if not all_vars:
                return ""

            return_vars = ", ".join(sorted(all_vars))
            query_parts.append(f"RETURN {return_vars}")
            query_parts.append("LIMIT 30")

            return "\n".join(query_parts)
        except Exception:
            return ""

    # ---------------------------------------------------------
    # Stage 3: Answer Generation
    # ---------------------------------------------------------
    def stage3_generate_answer(self, question: str, cypher: str, results: list) -> str:
        """Generate answer for a single query."""
        query_details = f"Cypher Query: {cypher}\n\nQuery Results: "
        results_str = json.dumps(results[:50], default=str, ensure_ascii=False)
        if len(results_str) > 3000:
            results_str = results_str[:3000] + "... (truncated)"
        query_details += results_str

        response = self.answer_chain.invoke({
            "question": question,
            "query_details": query_details,
        })
        return response.content.strip()

    def _stage3_generate_compound_answer(self, question: str, sub_results: list) -> str:
        """Generate a unified answer from multiple sub-query results."""
        print("\n--- Stage 3: Compound Answer Generation ---")
        query_details = ""
        for i, sr in enumerate(sub_results):
            query_details += f"\n--- Sub-question {i+1}: {sr['question']} ---\n"
            query_details += f"Cypher: {sr['cypher']}\n"
            results_str = json.dumps(sr["results"][:30], default=str, ensure_ascii=False)
            if len(results_str) > 1500:
                results_str = results_str[:1500] + "... (truncated)"
            query_details += f"Results: {results_str}\n"

        response = self.answer_chain.invoke({
            "question": question,
            "query_details": query_details,
        })
        answer = response.content.strip()
        print(f"\nAnswer: {answer}")
        return answer

    # ---------------------------------------------------------
    # Full Pipeline (compound-aware)
    # ---------------------------------------------------------
    def query(self, question: str) -> dict:
        """Run the full pipeline with automatic query decomposition."""
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

        sub_questions = self.stage1a_decompose(question)
        is_compound = len(sub_questions) > 1

        if not is_compound:
            cypher = self.stage1b_generate_cypher(question)
            results = self.stage2_execute_query(cypher)
            subgraph = self.stage2_extract_subgraph(cypher)
            answer = self.stage3_generate_answer(question, cypher, results)
            return {
                "question": question,
                "is_compound": False,
                "cypher": cypher,
                "results": results,
                "subgraph": subgraph,
                "answer": answer,
                "sub_results": [],
            }
        else:
            all_sub_results = []
            merged_subgraph = {"nodes": [], "edges": []}
            all_cyphers = []

            for sq in sub_questions:
                cypher = self.stage1b_generate_cypher(sq)
                results = self.stage2_execute_query(cypher)
                subgraph = self.stage2_extract_subgraph(cypher)
                merged_subgraph = self._merge_subgraphs(merged_subgraph, subgraph)
                all_cyphers.append(cypher)
                all_sub_results.append({
                    "question": sq,
                    "cypher": cypher,
                    "results": results,
                })

            answer = self._stage3_generate_compound_answer(question, all_sub_results)
            all_results = []
            for sr in all_sub_results:
                all_results.extend(sr["results"])

            return {
                "question": question,
                "is_compound": True,
                "cypher": "\n\n-- Sub-query --\n".join(all_cyphers),
                "results": all_results,
                "subgraph": merged_subgraph,
                "answer": answer,
                "sub_results": all_sub_results,
            }

    def close(self):
        self.driver.close()


# =============================================================
# Demo
# =============================================================
if __name__ == "__main__":
    pipeline = GraphRAGPipeline()

    demo_questions = [
        "What diagnoses does patient 10014729 have?",
        "What medications were prescribed for patient 10014729 and what were their diagnoses?",
        "How many abnormal lab results does patient 10014729 have?",
    ]

    for q in demo_questions:
        result = pipeline.query(q)
        print(f"\n{'='*60}")
        print(f"Compound: {result['is_compound']}")
        print(f"Answer: {result['answer']}")
        print("-" * 60)

    pipeline.close()