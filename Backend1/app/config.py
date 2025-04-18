# config.py

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Paths
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
RAG_DATA_PATH = os.getenv("RAG_DATA_PATH", "./data/rag_data")

# Model settings
DEFAULT_LLM_MODEL = "llama3-70b-8192"

# Create necessary directories
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(RAG_DATA_PATH, exist_ok=True)