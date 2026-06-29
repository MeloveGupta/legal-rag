"""
Legal RAG — FastAPI application entry point.

Run locally:
    uvicorn main:app --reload --port 8000

Production:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

Why workers=1?
The embedding model and BM25 index are loaded once per process.
Multiple workers would load them multiple times, multiplying memory use.
For scaling, run multiple containers behind a load balancer instead.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.legal_rag.api.routes import router
from src.legal_rag.config.settings import NVIDIA_MODEL_NAME, EMBEDDING_MODEL_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-warm all models when the server starts.

    Without this, the first user query triggers model loading
    which takes 20-30 seconds. Pre-warming means the server
    is fully ready before it accepts any traffic.
    """
    logger.info("=" * 50)
    logger.info("Legal RAG API starting up...")
    logger.info(f"LLM            : {NVIDIA_MODEL_NAME}")
    logger.info(f"Embeddings     : {EMBEDDING_MODEL_NAME}")
    logger.info("=" * 50)

    # Load embedding model into memory (downloads on first run)
    logger.info("Pre-warming embedding model...")
    from src.legal_rag.retrieval.embedder import get_embeddings
    get_embeddings()
    logger.info("Embedding model ready.")

    # Build BM25 index from ChromaDB
    logger.info("Building BM25 index...")
    from src.legal_rag.retrieval.bm25 import get_bm25_index
    index = get_bm25_index()
    logger.info(f"BM25 index ready: {len(index.documents)} documents.")

    # Load cross-encoder reranker
    logger.info("Loading cross-encoder reranker...")
    from src.legal_rag.retrieval.reranker import _get_cross_encoder
    _get_cross_encoder()
    logger.info("Cross-encoder ready.")

    logger.info("All models loaded. API is ready to serve requests.")
    logger.info("=" * 50)

    yield   # Server is running - handle requests

    logger.info("Legal RAG API shutting down.")


app = FastAPI(
    title="Legal RAG API",
    description=(
        "A retrieval-augmented generation system for Indian law. "
        "Every answer is grounded in retrieved source documents with citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allows your frontend (any domain during dev, specific domain in prod)
# to call this API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Tighten this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)