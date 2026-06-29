import logging
import time

from fastapi import APIRouter, HTTPException

from src.legal_rag.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceSchema,
    StatsResponse,
)
from src.legal_rag.config.settings import (
    EMBEDDING_MODEL_NAME,
    NVIDIA_MODEL_NAME,
)
from src.legal_rag.generation.generator import answer_query
from src.legal_rag.retrieval.vector_store import get_collection_stats

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Confirms the API is running and shows which models are loaded.
    Your deployment platform will ping this to know the service is alive.
    """
    return HealthResponse(
        status="ok",
        model=NVIDIA_MODEL_NAME,
        embedding_model=EMBEDDING_MODEL_NAME,
    )


@router.get("/stats", response_model=StatsResponse)
def collection_stats():
    """
    Returns how many chunks are in the vector store.
    Useful for verifying ingestion worked correctly.
    """
    try:
        stats = get_collection_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve stats.")


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    The main endpoint. Takes a legal question and returns
    a cited answer drawn from the ingested documents.

    If the question is out of scope, answered=False is returned
    with a clean refusal message. No hallucination, no guessing.
    """
    logger.info(f"Query received: '{request.query[:80]}'")
    start = time.time()

    try:
        response = answer_query(
            query=request.query,
            k=request.k,
            prompt_version=request.prompt_version,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query.",
        )

    latency_ms = int((time.time() - start) * 1000)
    logger.info(
        f"Query complete in {latency_ms}ms | "
        f"answered={response.answered} | "
        f"sources={len(response.sources)}"
    )

    sources = [
        SourceSchema(
            file_name=src.file_name,
            source=src.source,
            page=src.page,
            chunk_index=src.chunk_index,
            total_chunks=src.total_chunks,
            rerank_score=round(src.rerank_score, 4),
            content_preview=src.content_preview,
        )
        for src in response.sources
    ]

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        answered=response.answered,
        sources=sources,
        prompt_version=response.prompt_version,
        latency_ms=latency_ms,
    )