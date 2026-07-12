import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
            f" - Page {chunk.metadata.get('page', 'N/A')}"
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

_REASONING_MODELS = {
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-nano-30b-a3b",
}


def _get_llm() -> ChatOpenAI:
    kwargs = dict(
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        model=NVIDIA_MODEL_NAME,
        temperature=0,
        max_tokens=1024,
    )
    if NVIDIA_MODEL_NAME in _REASONING_MODELS:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return ChatOpenAI(**kwargs)

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
    """
    Detect whether the model refused to answer.

    Uses three layers in order:
    1. Exact configured refusal phrase
    2. Pattern matching on common refusal variants
    3. Heuristic: no citation + short-to-medium length answer
    """
    import re
    text_lower = answer_text.lower().strip()

    # Layer 1: exact configured phrase
    if refusal_phrase.lower() in text_lower:
        return True

    # Layer 2: common refusal variants across model sizes and temperatures
    refusal_patterns = [
        "cannot answer based on the provided",
        "cannot answer this question based on the provided",
        "not able to answer based on",
        "information is not available in the provided",
        "context does not contain",
        "context chunks do not contain",
        "chunks do not contain",
        "documents do not contain",
        "provided sources do not",
        "provided context does not",
        "not present in the provided",
        "not found in the provided",
        "context does not support",
        "the context does not support",
        "no information in the provided",
        "does not contain information about",
        "does not contain the specific",
        "i refuse",
    ]
    if any(pattern in text_lower for pattern in refusal_patterns):
        return True

    # Layer 3: no citation + no meaningful content
    # Every legitimate answer must cite at least one source [N].
    # Threshold raised to 250 chars to catch longer refusal sentences.
    has_citation = bool(re.search(r'\[\d+\]', answer_text))
    if not has_citation and len(answer_text.strip()) < 250:
        return True

    return False

def _detect_document_filter(query: str) -> Optional[str]:
    """
    Detect if the query is about a specific act and return its file_name.

    Why source filtering?
    When multiple acts are in the corpus, BNSS (which constantly references
    BNS section numbers in its schedules) dominates retrieval for BNS queries.
    When a query clearly names a specific act, restricting retrieval to that
    document eliminates cross-document noise and surfaces the right chunks.

    Order matters: check BNSS before BNS to avoid matching "bns" in "bnss".
    Returns None if no specific act is detected -> search all documents.
    """
    q = query.lower()

    # BNSS - check before BNS to avoid substring match
    if any(term in q for term in [
        "bharatiya nagarik suraksha sanhita", "bnss",
        "nagarik suraksha", "code of criminal procedure",
    ]):
        return "bnss.pdf"

    # BNS - avoid matching "bnss" by checking after BNSS
    if any(term in q for term in [
        "bharatiya nyaya sanhita", "nyaya sanhita",
    ]) or " bns " in f" {q} ":
        return "bns.pdf"

    # BSA
    if any(term in q for term in [
        "bharatiya sakshya", "sakshya adhiniyam",
    ]) or " bsa " in f" {q} ":
        return "bsa.pdf"

    # Constitution
    if any(term in q for term in [
        "constitution", "article ", "fundamental rights",
        "directive principles", "preamble",
    ]):
        return "constitution_of_india.pdf"
    
    # Labour Codes
    if any(term in q for term in ["code on wages", "wage code"]):
        return "code_on_wages_2019.pdf"

    if "industrial relations code" in q:
        return "industrial_relations_code_2020.pdf"

    if any(term in q for term in ["social security code", "code on social security"]):
        return "social_security_code_2020.pdf"

    if any(term in q for term in ["occupational safety", "osh code", "health and working conditions"]):
        return "osh_code_2020.pdf"
    
    # Civil law foundations
    if any(term in q for term in ["contract act", "indian contract act"]):
        return "contract_act_1872.pdf"

    if any(term in q for term in [
        "civil procedure code", "code of civil procedure",
    ]) or " cpc " in f" {q} ":
        return "civil_procedure_code_1908.pdf"

    if "transfer of property" in q:
        return "transfer_of_property_act_1882.pdf"

    if "specific relief" in q:
        return "specific_relief_act_1963.pdf"

    if "limitation act" in q:
        return "limitation_act_1963.pdf"

    if "general clauses act" in q:
        return "general_clauses_act_1897.pdf"
    
    # High practical utility
    if "motor vehicle" in q:
        return "motor_vehicles_act_1988.pdf"

    if any(term in q for term in [
        "negotiable instruments act", "cheque bounce",
        "dishonour of cheque", "bounced cheque",
    ]):
        return "negotiable_instruments_act_1881.pdf"

    if any(term in q for term in [
        "information technology act", "cyber crime", "cyber offence",
    ]):
        return "it_act_2000.pdf"

    if "consumer protection" in q:
        return "consumer_protection_act_2019.pdf"

    if any(term in q for term in ["right to information act", "rti act"]) or " rti " in f" {q} ":
        return "rti_act_2005.pdf"

    if any(term in q for term in [
        "digital personal data protection", "dpdp act", "data protection act",
    ]):
        return "dpdp_act_2023.pdf"

    return None   # No filter -> search all documents

def answer_query(
    query: str,
    k: int = TOP_K_RESULTS,
    prompt_version: str = "v2",
) -> RAGResponse:
    """
    Full RAG pipeline: hybrid search -> rerank -> LLM -> cited answer.
    """
    logger.info(f"Query: '{query[:80]}'")

    config = _load_prompt_config(prompt_version)

    fetch_k = max(k * 2, 10)
    file_filter = _detect_document_filter(query)
    if file_filter:
        logger.info(f"Source filter detected: {file_filter}")
    candidates = hybrid_search(
        query,
        k=fetch_k,
        fetch_k=fetch_k,
        file_name_filter=file_filter,
    )

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

    prompt = _build_prompt(query, chunks, config)
    llm    = _get_llm()

    logger.info(f"Calling {NVIDIA_MODEL_NAME}...")
    response    = llm.invoke(prompt)
    answer_text = response.content.strip()

    # Safety net: strip any <think>...</think> block that slips through
    import re
    answer_text = re.sub(r'<think>.*?</think>', '', answer_text, flags=re.DOTALL).strip()

    # Replace em dashes and en dashes from source legal text with spaced hyphens.
    answer_text = answer_text.replace('\u2014', ' - ').replace('\u2013', ' - ')

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