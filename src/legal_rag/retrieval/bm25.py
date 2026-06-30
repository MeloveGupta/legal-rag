import logging
import re
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.split(r'[\s\.,;:()\[\]\'\"]+', text)
    return [t for t in tokens if len(t) > 1]


class BM25Index:

    def __init__(self, documents: List[Document]):
        if not documents:
            raise ValueError(
            )

        self.documents = documents
        self.tokenized_corpus = [
            _tokenize(doc.page_content) for doc in documents
        ]
        self.index = BM25Okapi(self.tokenized_corpus)
        logger.info(
            f"BM25 index built with {len(documents)} documents."
        )

    def search(
    self,
    query: str,
    k: int = 10,
    file_name_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents, optionally filtered to a specific file.

        When file_name_filter is set, we fetch 3x more candidates before
        filtering so we still return k results after the filter is applied.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning("BM25 received empty query after tokenization.")
            return []

        scores = self.index.get_scores(query_tokens)

        # Fetch extra candidates when filtering so we have enough after
        fetch_n = k * 3 if file_name_filter else k
        top_indices = np.argsort(scores)[::-1][:fetch_n]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc = self.documents[idx]
            if file_name_filter and doc.metadata.get("file_name") != file_name_filter:
                continue
            results.append((doc, score))
            if len(results) >= k:
                break

        logger.info(
            f"BM25: {len(results)} results "
            f"(filter={file_name_filter or 'none'})"
        )
        return results


def _fetch_all_documents_from_store() -> List[Document]:
    from src.legal_rag.retrieval.vector_store import get_vector_store

    vector_store = get_vector_store()
    result = vector_store._collection.get(
        include=["documents", "metadatas"]
    )

    documents = []
    for content, metadata in zip(result["documents"], result["metadatas"]):
        documents.append(Document(
            page_content=content,
            metadata=metadata or {},
        ))

    logger.info(
        f"Fetched {len(documents)} documents from ChromaDB for BM25 indexing."
    )
    return documents


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    logger.info("Building BM25 index from ChromaDB corpus...")
    documents = _fetch_all_documents_from_store()
    return BM25Index(documents)


def bm25_search(
    query: str,
    k: int = 10,
    file_name_filter: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """Public interface for BM25 search."""
    index = get_bm25_index()
    return index.search(query, k=k, file_name_filter=file_name_filter)