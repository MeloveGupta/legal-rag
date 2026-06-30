import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.legal_rag.ingestion.chunker import chunk_documents
from src.legal_rag.ingestion.loader import load_documents
from src.legal_rag.retrieval.vector_store import (
    add_documents,
    get_collection_stats,
    reset_collection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_file(path: Path) -> int:
    """
    Ingest a single file into ChromaDB.
    Returns the number of chunks stored.
    """
    logger.info(f"Ingesting: {path.name}")
    docs   = load_documents(path)
    chunks = chunk_documents(docs)
    add_documents(chunks)
    logger.info(f"Done: {path.name} → {len(chunks)} chunks")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest legal documents into ChromaDB")
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific file paths to ingest",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all PDFs and Markdown files in data/raw/",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the vector store before ingesting",
    )
    args = parser.parse_args()

    if not args.files and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.reset:
        logger.warning("Resetting vector store - all existing chunks will be deleted.")
        reset_collection()

    # Collect files to ingest
    if args.all:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        files = sorted(
            list(raw_dir.glob("*.pdf")) +
            list(raw_dir.glob("*.md")) +
            list(raw_dir.glob("*.markdown"))
        )
        if not files:
            logger.error(f"No documents found in {raw_dir}")
            sys.exit(1)
        logger.info(f"Found {len(files)} document(s) to ingest.")
    else:
        files = [Path(f) for f in args.files]

    # Ingest
    total_chunks = 0
    for f in files:
        try:
            total_chunks += ingest_file(Path(f))
        except Exception as e:
            logger.error(f"Failed to ingest {f}: {e}")

    # Final stats
    stats = get_collection_stats()
    logger.info("=" * 50)
    logger.info(f"Ingestion complete.")
    logger.info(f"Chunks added this run : {total_chunks}")
    logger.info(f"Total chunks in store : {stats['total_chunks']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()