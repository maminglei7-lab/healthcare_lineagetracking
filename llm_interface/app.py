"""
Graph RAG Frontend - Streamlit Chat Interface
Run: streamlit run app.py
"""

import streamlit as st
from graph_rag import GraphRAGPipeline
from pyvis.network import Network
import streamlit.components.v1 as components
import json
import os
import tempfile

# =============================================================
# Node color scheme by label
# =============================================================
NODE_COLORS = {
    "Patient": "#4CAF50",
    "Admission": "#2196F3",
    "Diagnosis": "#FF9800",
    "Medication": "#E91E63",
    "LabTest": "#9C27B0",
    "ICD_Category": "#00BCD4",
}

def render_subgraph(subgraph: dict) -> str:
    """Render subgraph as an interactive pyvis HTML string with auto-fit."""
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    if not subgraph["nodes"]:
        return ""

    for node in subgraph["nodes"]:
        color = NODE_COLORS.get(node["label"], "#607D8B")
        display = node["display"]
        if len(display) > 40:
            display = display[:37] + "..."
        tooltip = f"<b>{node['label']}</b><br>"
        for k, v in list(node["properties"].items())[:6]:
            tooltip += f"{k}: {v}<br>"
        net.add_node(
            node["id"],
            label=display,
            title=tooltip,
            color=color,
            size=25 if node["label"] == "Patient" else 20,
            font={"size": 12},
        )

    for edge in subgraph["edges"]:
        tooltip = edge["type"]
        if edge["properties"]:
            props = ", ".join(f"{k}={v}" for k, v in list(edge["properties"].items())[:4])
            tooltip += f"\n{props}"
        net.add_edge(
            edge["source"],
            edge["target"],
            title=tooltip,
            label=edge["type"],
            font={"size": 9, "align": "middle"},
            arrows="to",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    net.save_graph(tmp.name)
    tmp.close()
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp.name)

    fit_js = """
    <style>
      #fit-btn {
        position: absolute; top: 10px; right: 10px; z-index: 9999;
        padding: 6px 14px; font-size: 13px; cursor: pointer;
        background: #4CAF50; color: white; border: none; border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
      }
      #fit-btn:hover { background: #388E3C; }
    </style>
    <button id="fit-btn" onclick="network.fit({animation: {duration: 500}})">⊞ Fit All</button>
    <script>
      network.once('stabilizationIterationsDone', function() {
        network.fit({animation: {duration: 800, easingFunction: 'easeInOutQuad'}});
      });
      setTimeout(function() { network.fit({animation: {duration: 500}}); }, 2000);
    </script>
    """
    html = html.replace("</body>", fit_js + "\n</body>")
    return html


def make_fullpage_html(subgraph: dict) -> str:
    """Generate a standalone full-page HTML for downloading."""
    net = Network(height="100vh", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    for node in subgraph["nodes"]:
        color = NODE_COLORS.get(node["label"], "#607D8B")
        display = node["display"]
        if len(display) > 50:
            display = display[:47] + "..."
        tooltip = f"<b>{node['label']}</b><br>"
        for k, v in list(node["properties"].items())[:6]:
            tooltip += f"{k}: {v}<br>"
        net.add_node(node["id"], label=display, title=tooltip, color=color,
                     size=30 if node["label"] == "Patient" else 22, font={"size": 14})

    for edge in subgraph["edges"]:
        tooltip = edge["type"]
        if edge["properties"]:
            props = ", ".join(f"{k}={v}" for k, v in list(edge["properties"].items())[:4])
            tooltip += f"\n{props}"
        net.add_edge(edge["source"], edge["target"], title=tooltip, label=edge["type"],
                     font={"size": 10, "align": "middle"}, arrows="to")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    net.save_graph(tmp.name)
    tmp.close()
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp.name)

    labels = set(n["label"] for n in subgraph["nodes"])
    legend_html = ""
    for label in sorted(labels):
        c = NODE_COLORS.get(label, "#607D8B")
        legend_html += f'<span style="color:{c};font-weight:bold;font-size:16px;">●</span> {label} &nbsp;&nbsp;'

    inject = f"""
    <div style="position:fixed;top:10px;left:10px;z-index:9999;background:rgba(255,255,255,0.9);padding:8px 16px;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,0.15);font-family:sans-serif;">
      {legend_html}
      <span style="color:#888;font-size:13px;">{len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges</span>
    </div>
    <script>
      network.once('stabilizationIterationsDone', function() {{
        network.fit({{animation: {{duration: 800}}}});
      }});
      setTimeout(function() {{ network.fit({{animation: {{duration: 500}}}}); }}, 2000);
    </script>
    """
    html = html.replace("</body>", inject + "\n</body>")
    html = html.replace("height: 600px;", "height: 100vh;")
    html = html.replace("<body>", '<body style="margin:0;padding:0;overflow:hidden;">')
    return html


# =============================================================
# Page Config
# =============================================================
st.set_page_config(page_title="Healthcare Graph RAG", page_icon="🏥", layout="wide")
st.title("🏥 Healthcare Graph RAG System")
st.caption("Intelligent Medical Query powered by Knowledge Graph + LLM")

@st.cache_resource
def init_pipeline():
    return GraphRAGPipeline()

pipeline = init_pipeline()

# =============================================================
# Sidebar
# =============================================================
with st.sidebar:
    st.header("📊 Graph Info")
    try:
        with pipeline.driver.session() as _s:
            _nodes = _s.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC").data()
            _rels = _s.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
            _total_nodes = sum(r["cnt"] for r in _nodes)
            _patient_cnt = next((r["cnt"] for r in _nodes if r["label"] == "Patient"), 0)
            st.markdown(f"- **Nodes**: {_total_nodes:,}")
            st.markdown(f"- **Relationships**: {_rels:,}")
            st.markdown(f"- **Patients**: {_patient_cnt:,}")
            with st.expander("Node breakdown"):
                for r in _nodes:
                    st.text(f"  {r['label']}: {r['cnt']:,}")
    except Exception as e:
        st.error(f"Failed to load graph stats: {e}")

    st.header("💡 Example Queries")
    example_queries = [
        "What diagnoses does patient 10014729 have?",
        "What medications were prescribed for patients with angina?",
        "How many abnormal lab results does patient 10014729 have?",
        "What are the most common diagnoses in the database?",
        "What medications were prescribed for patient 10014729 and what were their diagnoses?",
    ]
    for eq in example_queries:
        if st.button(eq, key=eq, use_container_width=True):
            st.session_state["input_query"] = eq

    st.divider()
    st.header("⚙️ Settings")
    from config import LLM_CONFIG, LLM_MODE
    st.text(f"LLM Mode: {LLM_MODE}")
    st.text(f"Model: {LLM_CONFIG[LLM_MODE]['model_name']}")

# =============================================================
# Chat History
# =============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "cypher" in msg:
            with st.expander("🔍 Generated Cypher"):
                st.code(msg["cypher"], language="cypher")
        if "subgraph_html" in msg and msg["subgraph_html"]:
            with st.expander("🕸️ Reasoning Subgraph"):
                components.html(msg["subgraph_html"], height=640, scrolling=True)
        if "record_count" in msg:
            st.caption(f"📦 Retrieved {msg['record_count']} records from Neo4j")

# =============================================================
# Chat Input
# =============================================================
chat_input = st.chat_input("Ask a medical question...")
user_input = st.session_state.pop("input_query", None) or chat_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔄 Processing..."):
            status = st.status("Running Graph RAG Pipeline...", expanded=True)

            # Stage 1a: Decompose question
            status.write("**Stage 1a:** Analyzing query complexity...")
            sub_questions = pipeline.stage1a_decompose(user_input)
            is_compound = len(sub_questions) > 1

            if is_compound:
                status.write(f"  ➜ Compound query detected — split into {len(sub_questions)} sub-queries")
                for i, sq in enumerate(sub_questions):
                    status.write(f"  &nbsp;&nbsp;&nbsp;[{i+1}] {sq}")

            # Execute each sub-question
            all_sub_results = []
            merged_subgraph = {"nodes": [], "edges": []}
            all_cyphers = []

            for i, sq in enumerate(sub_questions):
                prefix = f"[Sub-query {i+1}/{len(sub_questions)}] " if is_compound else ""

                status.write(f"**Stage 1b:** {prefix}Generating Cypher...")
                cypher = pipeline.stage1b_generate_cypher(sq)
                all_cyphers.append(cypher)

                status.write(f"**Stage 2:** {prefix}Querying Neo4j...")
                results = pipeline.stage2_execute_query(cypher)

                status.write(f"**Stage 2b:** {prefix}Extracting subgraph...")
                subgraph = pipeline.stage2_extract_subgraph(cypher)
                merged_subgraph = pipeline._merge_subgraphs(merged_subgraph, subgraph)

                all_sub_results.append({
                    "question": sq,
                    "cypher": cypher,
                    "results": results,
                })

            # Stage 3: Generate answer
            status.write("**Stage 3:** Generating evidence-based answer...")
            if is_compound:
                answer = pipeline._stage3_generate_compound_answer(user_input, all_sub_results)
            else:
                answer = pipeline.stage3_generate_answer(
                    user_input, all_cyphers[0], all_sub_results[0]["results"]
                )

            subgraph_html = render_subgraph(merged_subgraph) if merged_subgraph["nodes"] else ""
            combined_cypher = "\n\n-- Sub-query --\n".join(all_cyphers) if is_compound else all_cyphers[0]
            total_records = sum(len(sr["results"]) for sr in all_sub_results)

            status.update(label="✅ Pipeline complete!", state="complete", expanded=False)

            components.html("""
            <script>
                window.parent.document.querySelector('section.main').scrollTo({
                    top: window.parent.document.querySelector('section.main').scrollHeight,
                    behavior: 'smooth'
                });
            </script>
            """, height=0)

        # Display answer
        st.markdown(answer)

        # Show Cypher
        with st.expander("🔍 Generated Cypher"):
            if is_compound:
                for i, sr in enumerate(all_sub_results):
                    st.markdown(f"**Sub-query {i+1}:** {sr['question']}")
                    st.code(sr["cypher"], language="cypher")
            else:
                st.code(combined_cypher, language="cypher")

        # Show subgraph
        if subgraph_html:
            with st.expander("🕸️ Reasoning Subgraph", expanded=True):
                legend_items = []
                for label in sorted(set(n["label"] for n in merged_subgraph["nodes"])):
                    color = NODE_COLORS.get(label, "#607D8B")
                    legend_items.append(f'<span style="color:{color}; font-weight:bold;">●</span> {label}')
                st.markdown(" &nbsp;&nbsp; ".join(legend_items), unsafe_allow_html=True)

                col1, col2 = st.columns([5, 1])
                with col1:
                    st.caption(f"{len(merged_subgraph['nodes'])} nodes, {len(merged_subgraph['edges'])} edges")
                with col2:
                    fullpage = make_fullpage_html(merged_subgraph)
                    st.download_button(
                        label="📥 Download",
                        data=fullpage,
                        file_name="reasoning_subgraph.html",
                        mime="text/html",
                    )
                components.html(subgraph_html, height=640, scrolling=True)

        st.caption(f"📦 Retrieved {total_records} records from Neo4j")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "cypher": combined_cypher,
        "subgraph_html": subgraph_html,
        "record_count": total_records,
    })