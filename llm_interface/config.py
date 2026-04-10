"""
Graph RAG Pipeline Configuration
- LLM model switch: dev (cheap) vs prod (powerful)
- Neo4j connection settings loaded from .env
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# =============================================================
# LLM Configuration
# Switch MODE to change model:
#   "dev"  → gpt-3.5-turbo (cheap, for development & testing)
#   "prod" → gpt-4 (powerful, for final demo & presentation)
# =============================================================
LLM_MODE = "dev"

LLM_CONFIG = {
    "dev": {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0,        # 0 = deterministic, good for Cypher generation
        "max_tokens": 2048,
    },
    "prod": {
        "model_name": "gpt-4",
        "temperature": 0,
        "max_tokens": 4096,
    },
}

def get_llm_config():
    """Return current LLM configuration based on MODE."""
    return LLM_CONFIG[LLM_MODE]

# =============================================================
# Neo4j Configuration (loaded from .env)
# =============================================================
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# =============================================================
# OpenAI API Key (loaded from .env)
# =============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")