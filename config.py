"""
config.py - Central configuration for the Vietnamese Legal Assistant RAG System.

All configuration values are loaded from environment variables with sensible defaults.
This module is the single source of truth for all tunable parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(DATA_DIR / "faiss_index")))
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# API Keys
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================================
# Dataset Configuration
# ============================================================
DATASET_NAME = "th1nhng0/vietnamese-legal-documents"
DATASET_METADATA_CONFIG = "metadata"
DATASET_CONTENT_CONFIG = "content"
DATASET_SPLIT = "data"

# ============================================================
# Text Processing Configuration
# ============================================================
# Chunk size in characters (approximate token count * 4)
CHUNK_SIZE = 1024          # ~256 tokens per chunk (characters)
CHUNK_OVERLAP = 128        # Overlap between consecutive chunks
MIN_CHUNK_LENGTH = 50      # Minimum chunk length to keep (characters)
MAX_DOC_LENGTH = 500_000   # Skip documents longer than this (likely noise)

# ============================================================
# Embedding Configuration
# ============================================================
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIMENSION = 384   # Output dimension of the model above
EMBEDDING_BATCH_SIZE = 256  # Batch size for encoding
MAX_SEQ_LENGTH = 512        # Max token length for the embedding model

# ============================================================
# Milvus Configuration (Docker Standalone)
# ============================================================
# Connect to Milvus Standalone running in Docker at localhost:19530
# Start with: docker-compose up -d
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION_NAME = "legal_chunks"
# Milvus uses IVF_FLAT index with cosine similarity (metric_type="COSINE")
# IVF_FLAT partitions vectors into nlist clusters, then searches nprobe clusters
# This is ANN (Approximate Nearest Neighbor) — sub-linear search time
MILVUS_INDEX_TYPE = "IVF_FLAT"
MILVUS_METRIC_TYPE = "COSINE"   # Cosine similarity
MILVUS_NLIST = 128              # Number of IVF clusters
MILVUS_NPROBE = 16              # Number of clusters to search at query time

# ============================================================
# PostgreSQL Configuration (Docker — same server for legal_chunks + user data)
# ============================================================
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5433"))
PG_DB = os.getenv("PG_DB", "legal_db")
PG_USER = os.getenv("PG_USER", "legal_user")
PG_PASSWORD = os.getenv("PG_PASSWORD", "leducminh_2006")

# ============================================================
# Retrieval Configuration
# ============================================================
TOP_K = 3                   # Number of documents to retrieve
MIN_SIMILARITY_SCORE = 0.1  # Minimum cosine similarity threshold

# ============================================================
# RAG / LLM Configuration
# ============================================================
LLM_PROVIDER = "gemini"     # "gemini" or "openai"
GEMINI_MODEL = "gemini-2.0-flash"
OPENAI_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1       # Low temperature for factual answers
LLM_MAX_TOKENS = 2048       # Max output tokens

# ============================================================
# API Configuration
# ============================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ============================================================
# Logging Configuration
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
