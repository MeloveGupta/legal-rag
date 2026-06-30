import logging
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.legal_rag.config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    TOP_K_RESULTS,
)

from src.legal_rag.retrieval.embedder import get_embeddings

logger = logging.getLogger(__name__)

def get_vector_store() -> Chroma:
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    return vector_store

def add_documents(chunks: List[Document]) -> int:
    if not chunks:
        logger.warning("add_documents called with empty list. Nothing stored.")
        return 0
    
    logger.info(f"Embedding and storing {len(chunks)} chunks...")
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    logger.info(f"Successfully stores {len(chunks)} chunks in ChromaDB.")
    return len(chunks)

def similarity_search(
    query: str,
    k: int = TOP_K_RESULTS,
    file_name_filter: Optional[str] = None,
) -> List[Document]:
    """Retrieve top-k chunks, optionally filtered to a specific document."""
    vector_store = get_vector_store()
    where = {"file_name": file_name_filter} if file_name_filter else None
    results = vector_store.similarity_search(query, k=k, filter=where)
    logger.info(
        f"Vector search: {len(results)} results "
        f"(filter={file_name_filter or 'none'})"
    )
    return results


def similarity_search_with_scores(
    query: str,
    k: int = TOP_K_RESULTS,
    file_name_filter: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """Same as similarity_search but returns (Document, score) tuples."""
    vector_store = get_vector_store()
    where = {"file_name": file_name_filter} if file_name_filter else None
    results = vector_store.similarity_search_with_score(query, k=k, filter=where)
    return results

def get_collection_stats() -> dict:
    vector_store = get_vector_store()
    count = vector_store._collection.count()
    return {
        "collection_name": CHROMA_COLLECTION_NAME,
        "total_chunks": count,
        "db_path": CHROMA_DB_PATH,
    }

def reset_collection() -> None:
    vector_store = get_vector_store()
    vector_store.delete_collection()
    logger.warning(
        "Vector store collection deleted. "
        "All stored chunks have been removed."
    )