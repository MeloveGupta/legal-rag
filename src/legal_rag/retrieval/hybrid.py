import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.legal_rag.config.settings import TOP_K_RESULTS
from src.legal_rag.retrieval.bm25 import bm25_search
from src.legal_rag.retrieval.vector_store import similarity_search_with_scores

logger = logging.getLogger(__name__)

RRF_K = 60

def _get_chunk_id(doc: Document) -> str:
    meta = doc.metadata
    source = meta.get("source", "unknown")
    page= meta.get("page", 0)
    chunk_index = meta.get("chunk_index", 0)
    return f"{source}::page{page}::chunk{chunk_index}"

def reciprocal_rank_fusion(
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        k: int = RRF_K,
) -> List[Tuple[Document, float]]:
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, Document] = {}

    # score vector results - rank 1 is the lowest L2 distance(most similar)
    for rank, (doc, _score) in enumerate(vector_results, start=1):
        chunk_id = _get_chunk_id(doc)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_map[chunk_id] = doc

    # score bm25 results - rank 1 is the highest bm25 score(most relevant)
    for rank, (doc, _score) in enumerate(bm25_results, start=1):
        chunk_id = _get_chunk_id(doc)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_map[chunk_id] = doc

    # sort by rrf score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids]

def hybrid_search(
    query: str,
    k: int = TOP_K_RESULTS,
    fetch_k: int = 10,
    file_name_filter: Optional[str] = None,
) -> List[Document]:
    """
    Run hybrid retrieval: BM25 + vector search merged via RRF.

    file_name_filter restricts both search methods to a specific
    document when the query is clearly about one act only.
    """
    logger.info(
        f"Hybrid search: '{query[:60]}' "
        f"(filter={file_name_filter or 'none'})"
    )

    vector_results = similarity_search_with_scores(
        query,
        k=fetch_k,
        file_name_filter=file_name_filter,
    )
    bm25_results = bm25_search(
        query,
        k=fetch_k,
        file_name_filter=file_name_filter,
    )

    logger.info(
        f"Vector: {len(vector_results)} | BM25: {len(bm25_results)}"
    )

    fused  = reciprocal_rank_fusion(vector_results, bm25_results)
    top_k  = [doc for doc, _score in fused[:k]]
    return top_k
