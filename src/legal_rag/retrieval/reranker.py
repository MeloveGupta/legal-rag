import logging
from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.legal_rag.config.settings import TOP_K_RESULTS

logger  = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    logger.info(f"Loading cross-encoder reranker: {RERANKER_MODEL_NAME}")
    model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)
    logger.info("Cross-encoder loaded successfully.")
    return model

def rerank(
        query: str,
        documents: List[Document],
        top_k: int = TOP_K_RESULTS,
) -> List[Tuple[Document, float]]:
    if not documents:
        logger.warning("Reranker received empty document list.")
        return []
    
    model = _get_cross_encoder()

    # build(query, passage) pairs - one per candidate chunk
    pairs = [(query, doc.page_content) for doc in documents]

    logger.info(
        f"Reranking {len(pairs)} candidates for query: "
        f"'{query[:60]}{'...' if len(query) > 60 else ''}'"
    )

    # score all pairs in one batch - more efficient than one at a time
    scores = model.predict(pairs)

    # zip documents with their scores and sort descending
    scored_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    top_results = scored_docs[:top_k]

    logger.info(
        f"Reranking complete. "
        f"Top score: {top_results[0][1]:.4f} | "
        f"Bottom score: {top_results[-1][1]:.4f}"
    )

    return top_results