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


def _get_bge_token_count(text: str) -> int:
    """
    Count tokens using BGE's own tokenizer, not tiktoken.

    This is the count that actually determines whether a chunk gets
    silently truncated during embedding. tiktoken (used everywhere else
    in this file for chunk-size targeting) is the right tool for LLM
    prompt-budget math, but it is the wrong ruler for this specific
    question — BAAI/bge-large-en-v1.5 has a hard 512-token ceiling in
    its own BertTokenizerFast, and sentence-transformers truncates
    silently past that point with no error raised.
    """
    from src.legal_rag.retrieval.embedder import get_embeddings
    tokenizer = get_embeddings()._client.tokenizer
    return len(tokenizer.encode(text))


# BGE's real hard ceiling is 512. Leave margin for [CLS]/[SEP] special
# tokens the tokenizer adds automatically, and for normal variance in
# how dense legal text tokenizes compared to plain prose.
_BGE_SAFE_TOKEN_LIMIT = 480


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


# Matches the start of a new numbered legal provision — e.g.
# "14. Equality before law.—" or "103.(1) Whoever commits murder" or
# "11 . Res judicata.—" (space before the period). Handles optional
# single-letter suffixes ("21A.") and the period-space, period-openparen,
# and period-then-capital-letter formats seen across all 20 Acts in
# this corpus.
_SECTION_BOUNDARY_RE = re.compile(r'\n(?=\d{1,3}[A-Z]?\s?\.(?:\s|\(|[A-Z]))')

# Footnote/amendment-history markers use the identical "digit + period"
# format as real section numbers (e.g. "2. Ins. by s. 94, ibid." or
# "5. Clause (n) omitted by Act 7 of 2017") but describe the Act's own
# amendment history rather than starting a substantive provision. Left
# unfiltered, these get misread as section boundaries and corrupt
# whatever real section they land inside.
_FOOTNOTE_MARKER_RE = re.compile(
    r'^(?:Ins\.|Subs\.|Omitted|ibid|w\.e\.f\.|vide Notif|Rep\. by|Clause)',
    re.IGNORECASE,
)

# No hard-split block from section-boundary splitting may exceed this
# many tiktoken tokens, however confident the boundary detector was.
# Schedules, forms, and appendices at the end of an Act often use
# formatting the boundary regex was never designed to recognise —
# without this ceiling, everything after the last recognised section
# silently merges into one unbounded chunk. Observed in practice: a
# 49,000+ token BNSS chunk swallowing the entire First Schedule.
_MAX_SECTION_BLOCK_TOKENS_MULTIPLIER = 3


def _split_by_section_boundary(text: str) -> List[str]:
    matches = list(_SECTION_BOUNDARY_RE.finditer(text))

    real_positions = []
    for m in matches:
        pos = m.start()
        peek = text[pos:pos + 120].lstrip('\n')
        after_number = re.sub(r'^\d{1,3}[A-Z]?\s?\.\s*', '', peek)
        if _FOOTNOTE_MARKER_RE.match(after_number):
            continue  # footnote, not a real section start — skip it
        real_positions.append(pos)

    if not real_positions:
        return [text]

    blocks = []
    if real_positions[0] > 0:
        blocks.append(text[:real_positions[0]])
    for i, pos in enumerate(real_positions):
        end = real_positions[i + 1] if i + 1 < len(real_positions) else len(text)
        blocks.append(text[pos:end])
    return [b.strip() for b in blocks if b.strip()]


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

        # Hard split on section/article boundaries first — never merged
        # back together, regardless of how short an individual piece is.
        section_blocks = _split_by_section_boundary(full_text)
        raw_chunks: List[str] = []
        max_block_tokens = chunk_size * _MAX_SECTION_BLOCK_TOKENS_MULTIPLIER
        for block in section_blocks:
            if _get_token_count(block) <= max_block_tokens:
                raw_chunks.append(block)
            else:
                # Either a genuinely oversized section, or — more likely —
                # unrecognised trailing structure (schedules, forms) the
                # boundary regex couldn't segment. Either way, never let
                # a block through unbounded.
                raw_chunks.extend(splitter.split_text(block))

        logger.info(
            f"Document {doc_index + 1}/{len(docs_to_chunk)} "
            f"({doc.metadata.get('file_name', doc.metadata.get('source', 'unknown'))}) "
            f"→ {len(raw_chunks)} section-level blocks"
        )

        # Collected separately per document so chunk_index/total_chunks
        # can be numbered correctly AFTER the BGE safety-net pass below —
        # one raw_chunk can expand into several final chunks, and
        # total_chunks must reflect this document's true final count,
        # not a count taken before that expansion happens.
        doc_chunks: List[Document] = []

        cursor = 0
        for chunk_text in raw_chunks:
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

            # Final safety net, measured against BGE's own tokenizer, not
            # tiktoken. tiktoken is the right tool for LLM prompt-budget
            # math elsewhere in this file, but the wrong ruler for this
            # question specifically: BAAI/bge-large-en-v1.5 has a hard
            # 512-token ceiling in its own BertTokenizerFast, and
            # sentence-transformers truncates silently past that point
            # with no error raised. Every chunk is checked here regardless
            # of how it was produced upstream — section boundary, footnote
            # filtering, or the tiktoken-based fallback above.
            if _get_bge_token_count(clean_text) > _BGE_SAFE_TOKEN_LIMIT:
                bge_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=_BGE_SAFE_TOKEN_LIMIT,
                    chunk_overlap=50,
                    length_function=_get_bge_token_count,
                    separators=["\n\n", "\n", ".", " ", ""],
                )
                sub_pieces = bge_splitter.split_text(clean_text)
            else:
                sub_pieces = [clean_text]

            for sub_text in sub_pieces:
                doc_chunks.append(Document(
                    page_content=sub_text,
                    metadata={
                        **doc.metadata,
                        "token_count": _get_token_count(sub_text),
                        "page": page_num,
                    },
                ))

        # Now that this document's true final chunk count is known —
        # after BGE-driven expansion — number chunk_index/total_chunks
        # correctly, scoped to this document only.
        for i, c in enumerate(doc_chunks):
            c.metadata["chunk_index"] = i
            c.metadata["total_chunks"] = len(doc_chunks)

        logger.info(f"  → {len(doc_chunks)} final chunks after BGE safety-net pass")
        all_chunks.extend(doc_chunks)

    logger.info(f"Chunking complete: {len(documents)} page(s) → {len(all_chunks)} total chunks")
    return all_chunks