from typing import Literal

from pydantic import BaseModel


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
