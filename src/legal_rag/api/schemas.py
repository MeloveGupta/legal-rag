from pydantic import BaseModel, Field
from typing import List


class QueryRequest(BaseModel):
    """
    What the client sends to /api/query.
    """
    query: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The legal question to answer",
        examples=["What does Article 21 say about right to life?"],
    )
    k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of chunks to retrieve",
    )
    prompt_version: str = Field(
        default="v2",
        description="Which prompt version to use",
    )


class SourceSchema(BaseModel):
    """
    A single cited source chunk returned with the answer.
    """
    file_name: str
    source: str
    page: int
    chunk_index: int
    total_chunks: int
    rerank_score: float
    content_preview: str


class QueryResponse(BaseModel):
    """
    What the API returns for every query.
    """
    query: str
    answer: str
    answered: bool
    sources: List[SourceSchema]
    prompt_version: str
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_model: str


class StatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    db_path: str