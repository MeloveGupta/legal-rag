import bisect
import logging
import re
from collections import defaultdict
from typing import List, Tuple

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.legal_rag.config.settings import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


def _get_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def _merge_page_documents(documents: List[Document]) -> List[Document]:
    groups: dict = defaultdict(list)
    for doc in documents:
        groups[doc.metadata.get("source", "unknown")].append(doc)

    merged_docs = []
    for source, pages in groups.items():
        pages.sort(key=lambda d: d.metadata.get("page", 0))
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

    logger.info(f"Merged {len(documents)} pages from {len(merged_docs)} source file(s)")
    return merged_docs


def _build_page_offset_index(full_text: str) -> List[Tuple[int, int]]:
    return [
        (m.start(), int(m.group(1)))
        for m in re.finditer(r'\[PAGE (\d+)\]', full_text)
    ]


def _page_for_offset(offset_index: List[Tuple[int, int]], char_offset: int) -> int:
    if not offset_index:
        return 1
    offsets_only = [o for o, _ in offset_index]
    idx = bisect.bisect_right(offsets_only, char_offset) - 1
    return offset_index[max(idx, 0)][1]


def _clean_page_markers(text: str) -> str:
    return re.sub(r'\[PAGE \d+\]\n?', '', text).strip()


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    merge_pages: bool = True,
) -> List[Document]:
    is_pdf = any(doc.metadata.get("file_type") == "pdf" for doc in documents)
    docs_to_chunk = _merge_page_documents(documents) if (merge_pages and is_pdf) else documents

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_chunks: List[Document] = []

    for doc_index, doc in enumerate(docs_to_chunk):
        full_text = doc.page_content
        offset_index = _build_page_offset_index(full_text)
        raw_chunks = splitter.split_text(full_text)
        total_chunks = len(raw_chunks)

        logger.info(
            f"Document {doc_index + 1}/{len(docs_to_chunk)} "
            f"({doc.metadata.get('file_name', doc.metadata.get('source', 'unknown'))}) "
            f"→ {total_chunks} chunks"
        )

        cursor = 0
        for chunk_index, chunk_text in enumerate(raw_chunks):
            # Locate this chunk's true position in the original text using
            # a prefix as the search key — the splitter strips outer
            # whitespace from chunks, so an exact full-text match can fail.
            search_key = chunk_text[:80]
            start = full_text.find(search_key, cursor)
            if start == -1:
                start = full_text.find(search_key)
            if start == -1:
                start = cursor

            page_num = _page_for_offset(offset_index, start)
            cursor = start + 1   # keep subsequent searches moving forward

            clean_text = _clean_page_markers(chunk_text)

            all_chunks.append(Document(
                page_content=clean_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "token_count": _get_token_count(clean_text),
                    "page": page_num,
                },
            ))

    logger.info(f"Chunking complete: {len(documents)} page(s) → {len(all_chunks)} total chunks")
    return all_chunks