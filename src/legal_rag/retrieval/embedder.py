import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.legal_rag.config.settings import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    logger.info("First run downloads ~1.3GB. Subsequent runs use local cache.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model loaded successfully.")
    return embeddings