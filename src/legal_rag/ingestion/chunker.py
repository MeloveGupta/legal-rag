import logging
import re
from collections import defaultdict
from typing import List

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.legal_rag.config.settings import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


def _get_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def _merge_page_documents(documents: List[Document]) -> List[Document]:
    # group pages by their source file
    groups: dict = defaultdict(list)
    for doc in documents:
        groups[doc.metadata.get("source", "unknown")].append(doc)

    merged_docs = []
    for source, pages in groups.items():
        # sort pages by page number to ensure correct order
        pages.sort(key=lambda d: d.metadata.get("page", 0))

        # concatenate with page markers
        full_text = ""
        for page in pages:
            page_num = page.metadata.get("page", "?")
            full_text += f"\n\n[PAGE {page_num}]\n{page.page_content.strip()}"

        merged_docs.append(Document(
            page_content=full_text.strip(),
            metadata={
                **pages[0].metadata,
                "total_pages": len(pages),
                "page": 1,
            }
        ))

    logger.info(
        f"Merged {len(documents)} pages from "
        f"{len(merged_docs)} source file(s)"
    )
    return merged_docs


def _extract_page_from_chunk(text: str) -> int:
    markers = re.findall(r'\[PAGE (\d+)\]', text)
    if markers:
        return int(markers[-1])
    return 1


def _clean_page_markers(text: str) -> str:
    return re.sub(r'\[PAGE \d+\]\n?', '', text).strip()


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    merge_pages: bool = True,
) -> List[Document]:
    is_pdf = any(
        doc.metadata.get("file_type") == "pdf"
        for doc in documents
    )

    # Merge pages only for PDFs - web/markdown are already single documents
    if merge_pages and is_pdf:
        docs_to_chunk = _merge_page_documents(documents)
    else:
        docs_to_chunk = documents

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ".",
            " ",
            "",
        ],
    )

    all_chunks: List[Document] = []

    for doc_index, doc in enumerate(docs_to_chunk):
        raw_chunks = splitter.split_documents([doc])
        total_chunks = len(raw_chunks)

        logger.info(
            f"Document {doc_index + 1}/{len(docs_to_chunk)} "
            f"({doc.metadata.get('file_name', doc.metadata.get('source', 'unknown'))}) "
            f"→ {total_chunks} chunks"
        )

        for chunk_index, chunk in enumerate(raw_chunks):
            # Extract page number from [PAGE N] markers before cleaning
            page_num = _extract_page_from_chunk(chunk.page_content)

            # Clean markers from text - LLM sees clean content
            chunk.page_content = _clean_page_markers(chunk.page_content)

            chunk.metadata.update({
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "token_count": _get_token_count(chunk.page_content),
                "page": page_num,
            })
            all_chunks.append(chunk)

    logger.info(
        f"Chunking complete: {len(documents)} page(s) "
        f"→ {len(all_chunks)} total chunks"
    )
    return all_chunks