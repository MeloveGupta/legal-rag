import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from src.legal_rag.config.settings import (
    NVIDIA_API_KEY,
    NVIDIA_MODEL_NAME,
    NVIDIA_BASE_URL,
    TOP_K_RESULTS,
    ROOT_DIR,
)
from src.legal_rag.retrieval.hybrid import hybrid_search
from src.legal_rag.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

RERANK_RELEVANCE_THRESHOLD = -5.0


def _load_prompt_config(version: str = "v1") -> dict:
    prompt_path = ROOT_DIR / "prompts" / version / "qa.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt config not found at {prompt_path}")
    with open(prompt_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded prompt: {config['name']} v{config['version']}")
    return config


@dataclass
class CitedSource:
    file_name: str
    source: str
    page: int
    chunk_index: int
    total_chunks: int
    token_count: int
    content_preview: str
    rerank_score: float = 0.0


@dataclass
class RAGResponse:
    answer: str
    sources: List[CitedSource]
    query: str
    retrieved_chunks: List[Document]
    answered: bool = True
    prompt_version: str = "v1"


def _build_prompt(query: str, chunks: List[Document], config: dict) -> str:
    context_block = ""
    for i, chunk in enumerate(chunks):
        source_label = (
            f"{chunk.metadata.get('file_name', chunk.metadata.get('source', 'Unknown'))}"
            f" — Page {chunk.metadata.get('page', 'N/A')}"
            f" (Chunk {chunk.metadata.get('chunk_index', i) + 1}"
            f" of {chunk.metadata.get('total_chunks', '?')})"
        )
        context_block += f"\n[{i+1}] SOURCE: {source_label}\n"
        context_block += f"{chunk.page_content.strip()}\n"
        context_block += "-" * 60 + "\n"

    rules = config.get("rules", [])
    rules_formatted = "\n".join(
        f"{i+1}. {rule}" for i, rule in enumerate(rules)
    )
    return config["template"].format(
        system=config["system"].strip(),
        rules_formatted=rules_formatted,
        context_block=context_block,
        query=query,
    )


def _get_llm() -> ChatOpenAI:
    """
    NVIDIA NIM uses an OpenAI-compatible API.
    Pointing langchain-openai at NVIDIA's endpoint with the
    Nemotron Ultra model requires only base_url and api_key changes.
    temperature=0 - non-negotiable for legal systems.
    """
    return ChatOpenAI(
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        model=NVIDIA_MODEL_NAME,
        temperature=0,
        max_tokens=1024,
    )


def _extract_sources(
    chunks: List[Document],
    scores: List[float],
) -> List[CitedSource]:
    sources = []
    for chunk, score in zip(chunks, scores):
        meta = chunk.metadata
        sources.append(CitedSource(
            file_name=meta.get("file_name", meta.get("source", "Unknown")),
            source=meta.get("source", "Unknown"),
            page=meta.get("page", 1),
            chunk_index=meta.get("chunk_index", 0),
            total_chunks=meta.get("total_chunks", 1),
            token_count=meta.get("token_count", 0),
            content_preview=chunk.page_content[:200].strip(),
            rerank_score=float(score),
        ))
    return sources

def _is_refusal(answer_text: str, refusal_phrase: str) -> bool:
    text_lower = answer_text.lower().strip()

    # Primary: check exact configured phrase
    if refusal_phrase.lower() in text_lower:
        return True

    # Secondary: common refusal patterns (model may paraphrase)
    refusal_patterns = [
        "cannot answer based on the provided",
        "cannot answer this question based on the provided",
        "not able to answer based on",
        "information is not available in the provided",
        "context does not contain",
        "documents do not contain",
        "provided sources do not",
        "not present in the provided",
        "the context does not support",
        "no information in the provided",
        "not found in the provided",
        "i refuse",
    ]
    if any(pattern in text_lower for pattern in refusal_patterns):
        return True

    # tertiary: a short answer with no citation markers is a refusal
    # every legitimate answer MUST cite at least one source [N]
    import re
    has_citation = bool(re.search(r'\[\d+\]', answer_text))
    if not has_citation and len(answer_text.strip()) < 120:
        return True

    return False

def answer_query(
    query: str,
    k: int = TOP_K_RESULTS,
    prompt_version: str = "v2",
) -> RAGResponse:
    """
    Full RAG pipeline: hybrid search -> rerank -> Nemotron Ultra -> cited answer.
    """
    logger.info(f"Query: '{query[:80]}'")

    config = _load_prompt_config(prompt_version)

    fetch_k = max(k * 2, 10)
    candidates = hybrid_search(query, k=fetch_k, fetch_k=fetch_k)

    if not candidates:
        return RAGResponse(
            answer="I cannot answer this question as no relevant documents were found.",
            sources=[], query=query, retrieved_chunks=[],
            answered=False, prompt_version=prompt_version,
        )

    reranked   = rerank(query, candidates, top_k=k)
    chunks     = [doc   for doc,  _score in reranked]
    scores     = [score for _doc, score  in reranked]
    best_score = scores[0] if scores else float("-inf")

    if best_score < RERANK_RELEVANCE_THRESHOLD:
        logger.info(f"Rerank threshold: {best_score:.4f} < {RERANK_RELEVANCE_THRESHOLD}. Refusing.")
        return RAGResponse(
            answer="I cannot answer this question based on the available documents.",
            sources=_extract_sources(chunks, scores),
            query=query, retrieved_chunks=chunks,
            answered=False, prompt_version=prompt_version,
        )

    prompt   = _build_prompt(query, chunks, config)
    llm      = _get_llm()

    logger.info(f"Calling {NVIDIA_MODEL_NAME}...")
    response    = llm.invoke(prompt)
    answer_text = response.content.strip()

    refusal_phrase = config.get(
        "refusal_phrase",
        "I cannot answer this question based on the available documents"
    )
    answered = not _is_refusal(answer_text, refusal_phrase)

    return RAGResponse(
        answer=answer_text,
        sources=_extract_sources(chunks, scores),
        query=query, retrieved_chunks=chunks,
        answered=answered, prompt_version=prompt_version,
    )