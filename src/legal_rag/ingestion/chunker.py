import logging
from typing import List

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.legal_rag.config.settings import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

def _get_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def chunk_documents(
        documents: List[Document],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
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

    for doc_index, doc in enumerate(documents):
        raw_chunks = splitter.split_documents([doc])
        total_chunks = len(raw_chunks)

        logger.info(
            f"Document {doc_index + 1}/{len(documents)} "
            f"({doc.metadata.get('file_name', doc.metadata.get('source', 'unknown'))}) "
            f"-> {total_chunks} chunks"
        )

        for chunk_index, chunk in enumerate(raw_chunks):
            chunk.metadata.update({
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "token_count": _get_token_count(chunk.page_content),
            })
            all_chunks.append(chunk)

    logger.info(
        f"Chunking complete: {len(documents)} document(s) "
        f"-> {len(all_chunks)} total chunks"
    )
    return all_chunks