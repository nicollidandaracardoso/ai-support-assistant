from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["healthy"]


class IngestionResponse(BaseModel):
    documents_processed: int
    chunks_stored: int


class StoredDocument(BaseModel):
    document_id: str
    source: str
    category: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: list[StoredDocument]
    total: int


class DeleteDocumentResponse(BaseModel):
    document_id: str
    chunks_deleted: int


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    category: str | None = Field(default=None, min_length=1)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 3:
            raise ValueError("A pergunta deve ter pelo menos 3 caracteres úteis.")
        return normalized

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("A categoria não pode ficar vazia.")
        return normalized


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    category: str
    chunk_index: int
    content: str
    distance: float


class SearchResponse(BaseModel):
    question: str
    results: list[SearchResult]
    total: int
