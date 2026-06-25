import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from src.legal_rag.config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    TOP_K_RESULTS,
    ROOT_DIR,
)
from src.legal_rag.retrieval.hybrid import hybrid_search
from src.legal_rag.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

# prompt loading

def _load_prompt_config(version: str = "v1") -> dict:
    prompt_path = ROOT_DIR / "prompts" / version / "qa.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt config not found at {prompt_path}. "
            f"Make sure prompts/{version}/qa.yaml exists."
        )
    with open(prompt_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(
        f"Loaded prompt config: {config['name']} v{config['version']}"
    )
    return config


# data structures

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


# prompt builder

def _build_prompt(
    query: str,
    chunks: List[Document],
    config: dict,
) -> str:
    # build numbered context block
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

    # format rules as numbered list
    rules = config.get("rules", [])
    rules_formatted = "\n".join(
        f"{i+1}. {rule}" for i, rule in enumerate(rules)
    )

    prompt = config["template"].format(
        system=config["system"].strip(),
        rules_formatted=rules_formatted,
        context_block=context_block,
        query=query,
    )
    return prompt


# LLM setup

def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL_NAME,
        temperature=0,
        max_tokens=1024,
    )


# citation extractor

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


# main entry point

def answer_query(
    query: str,
    k: int = TOP_K_RESULTS,
    prompt_version: str = "v1",
) -> RAGResponse:
    logger.info(
        f"Processing query: '{query[:80]}{'...' if len(query) > 80 else ''}'"
    )

    # load prompt config
    config = _load_prompt_config(prompt_version)

    # hybrid retrieval - fetch more candidates than we need
    fetch_k = max(k * 2, 10)
    candidates = hybrid_search(query, k=fetch_k, fetch_k=fetch_k)

    if not candidates:
        logger.warning("No candidates retrieved from hybrid search.")
        return RAGResponse(
            answer="I cannot answer this question as no relevant documents were found.",
            sources=[],
            query=query,
            retrieved_chunks=[],
            answered=False,
            prompt_version=prompt_version,
        )

    # rerank candidates
    reranked = rerank(query, candidates, top_k=k)
    chunks = [doc for doc, _score in reranked]
    scores = [score for _doc, score in reranked]

    RERANK_RELEVANCE_THRESHOLD = -3.0
    best_score = scores[0] if scores else float("-inf")

    if best_score < RERANK_RELEVANCE_THRESHOLD:
        logger.info(
            f"Best rerank score {best_score:.4f} is below threshold "
            f"{RERANK_RELEVANCE_THRESHOLD}. Returning refusal without LLM call."
        )
        return RAGResponse(
            answer="I cannot answer this question based on the available documents.",
            sources=_extract_sources(chunks, scores),
            query=query,
            retrieved_chunks=chunks,
            answered=False,
            prompt_version=prompt_version,
        )

    logger.info(f"Using top {len(chunks)} reranked chunks for generation.")

    # build prompt and call LLM
    prompt = _build_prompt(query, chunks, config)
    llm = _get_llm()

    logger.info("Calling Groq LLM...")
    response = llm.invoke(prompt)
    answer_text = response.content.strip()

    # detect refusal
    refusal_phrase = config.get(
        "refusal_phrase",
        "I cannot answer this question based on the available documents"
    )
    answered = refusal_phrase.lower() not in answer_text.lower()

    if not answered:
        logger.info("LLM declined — insufficient evidence in retrieved chunks.")
    else:
        logger.info("Answer generated successfully with citations.")

    # package and return
    return RAGResponse(
        answer=answer_text,
        sources=_extract_sources(chunks, scores),
        query=query,
        retrieved_chunks=chunks,
        answered=answered,
        prompt_version=prompt_version,
    )