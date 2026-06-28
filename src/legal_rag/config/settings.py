import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# project paths
ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# NVIDIA NIM LLM
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL_NAME = os.getenv(
    "NVIDIA_MODEL_NAME",
    "nvidia/nemotron-3-ultra-550b-a55b"
)
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
)

# embeddings
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-large-en-v1.5"
)

# vector store
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(ROOT_DIR / "chroma_db"))
CHROMA_COLLECTION_NAME = "legal_documents"

# retrieval
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# validation
if not NVIDIA_API_KEY:
    raise EnvironmentError(
        "NVIDIA_API_KEY is not set. "
        "Get a free key at build.nvidia.com - no credit card needed."
    )