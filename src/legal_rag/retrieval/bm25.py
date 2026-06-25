import logging
import re
from functools import lru_cache
from typing import List, Tuple

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
    ) -> List[Tuple[Document, float]]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning("BM25 received empty query after tokenization.")
            return []

        scores = self.index.get_scores(query_tokens)

        # Get indices of top-k scores, sorted descending
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:   # Only return chunks with at least one term match
                results.append((self.documents[idx], score))

        logger.info(
            f"BM25 retrieved {len(results)} results "
            f"(with score > 0) for query: "
            f"'{query[:60]}{'...' if len(query) > 60 else ''}'"
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
) -> List[Tuple[Document, float]]:
    index = get_bm25_index()
    return index.search(query, k=k)