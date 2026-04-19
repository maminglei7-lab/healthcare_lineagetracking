# Healthcare Data Lineage Tracking

A three-person team project that builds a comprehensive system to track how healthcare data flows and transforms through processing pipelines, using MIMIC-IV demo data.

## Team

| Member | Responsibility |
|--------|---------------|
| Minglei | ETL Pipeline |
| Yining | Neo4j Knowledge Graph |
| Guangyi | LLM Interface |

## Project Structure

```
healthcare_lineagetracking/
├── etl/                  # ETL pipeline (Minglei)
│   ├── extract.py
│   ├── transform.py
│   ├── quality_check.py
│   ├── lineage_decorator.py
│   └── build_graph_input.py
├── knowledge_graph/      # Neo4j graph implementation (Guangyi)
├── llm_interface/        # LLM interface (YiNing)
├── data/
│   ├── raw/              # Raw MIMIC-IV input files (not uploaded)
│   ├── cleaned/          # Cleaned output from ETL
│   └── graph_input/      # Neo4j-ready node and relationship files
└── docs/                 # Documentation and meeting notes
```

## Data

This project uses [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) demo data (100 patients). The following source files are required in `data/raw/`:

- `patients.csv`
- `admissions.csv`
- `diagnoses_icd.csv`
- `labevents.csv`
- `d_icd_diagnoses.csv`
- `d_labitems.csv`

> Raw data files are not committed to this repository.

## ETL Pipeline

The ETL module processes raw MIMIC-IV data through the following steps:

1. **Extract** — Load raw CSV files
2. **Transform** — Clean and standardize data, generate derived fields
3. **Quality Check** — Validate data integrity
4. **Build Graph Input** — Generate Neo4j-compatible node and relationship files

All operations are automatically logged to `lineage.json` for full data lineage tracking.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/maminglei7-lab/healthcare_lineagetracking.git

# Install dependencies (TBD)
pip install -r requirements.txt

# Run ETL pipeline (TBD)
python etl/main.py
```

## Status

🚧 Project in progress
