import logging
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from src.legal_rag.config.settings import GROQ_API_KEY, GROQ_MODEL_NAME, TOP_K_RESULTS
from src.legal_rag.retrieval.vector_store import similarity_search

logger = logging.getLogger(__name__)

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

@dataclass
class RAGResponse:
    answer: str
    sources: List[CitedSource]
    query: str
    retrieved_chunks: List[Document]
    answered: bool = True

# prompt builder
def _build_prompt(query: str, chunks: List[Document]) -> str:
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
    prompt = f"""You are a legal research assistant specializing in Indian law.
    Your role is to answer questions accurately based ONLY on the provided source documents.

    STRICT RULES YOU MUST FOLLOW:
    1. Base your answer EXCLUSIVELY on the context provided below.
    2. Do NOT use any prior knowledge or external information.
    3. Every factual claim you make MUST be followed by a citation in the format [1], [2] etc.
    referring to the numbered sources below.
    4. If the provided context does not contain sufficient information to answer the question,
    respond ONLY with: "I cannot answer this question based on the available documents."
    Do not attempt to answer from general knowledge.
    5. Do not speculate, infer, or extrapolate beyond what is explicitly stated in the sources.
    6. If multiple sources support a claim, cite all of them: [1][3].

    CONTEXT - Retrieved source documents:
    {context_block}

    QUESTION:
    {query}

    ANSWER (with inline citations):"""

    return prompt

# llm setup
def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL_NAME,
        temperature=0,
        max_tokens=1024,
    )

# citation extractor
def _extract_sources(chunks: List[Document]) -> List[CitedSource]:
    sources = []
    for chunk in chunks:
        meta = chunk.metadata
        sources.append(CitedSource(
            file_name=meta.get("file_name", meta.get("source", "Unknown")),
            source=meta.get("source", "Unknown"),
            page=meta.get("page", 1),
            chunk_index=meta.get("chunk_index", 0),
            total_chunks=meta.get("total_chunks", 1),
            token_count=meta.get("token_count", 0),
            content_preview=chunk.page_content[:200].strip(),
        ))
        return sources
    
# main entry point
def answer_query(
        query: str,
        k: int = TOP_K_RESULTS,
) -> RAGResponse:
    logger.info(f"Processing query: '{query[:80]}{'...' if len(query) > 80 else ''}'")

    # Retrieve
    chunks = similarity_search(query, k=k)
    if not chunks:
        logger.warning("No chunks retrieved from vector store.")
        return RAGResponse(
            answer="I cannot answer this question as no relevant documents were found.",
            sources=[],
            query=query,
            retrieved_chunks=[],
            answered=False
        )
    
    logger.info(f"Retrieved {len(chunks)} chunks for generation.")

    # build prompt
    prompt = _build_prompt(query, chunks)

    llm = _get_llm()
    logger.info("Calling Groq LLM...")
    response = llm.invoke(prompt)
    answer_text = response.content.strip()

    # detect refusal from model
    refusal_phrase = "I cannot answer this question based on the available documents"
    answered = refusal_phrase.lower() not in answer_text.lower()

    if not answered:
        logger.info("LLM declined to answer, insufficient evidence in the retrieved chunks.")
    else:
        logger.info("Answer generated successfully with citations.")

    # package and return
    return RAGResponse(
        answer=answer_text,
        sources=_extract_sources(chunks),
        query=query,
        retrieved_chunks=chunks,
        answered=answered,
    )