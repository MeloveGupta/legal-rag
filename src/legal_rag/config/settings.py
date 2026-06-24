import os
from pathlib import Path
from dotenv import load_dotenv

# load .env from project root
load_dotenv()

# project paths
ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# groq llm
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

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
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Copy .env.example to .env and add your key"
    )