from fastapi import FastAPI, HTTPException

from app.document_processing import load_document_chunks
from app.providers.embeddings import EmbeddingProvider, GeminiEmbeddingProvider
from app.schemas import (
    DeleteDocumentResponse,
    DocumentsResponse,
    HealthResponse,
    IngestionResponse,
)
from app.vector_store import ChromaVectorStore, group_chunks_by_document

app = FastAPI(
    title="AI Support Assistant",
    description="API para um assistente de suporte baseado em documentação.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.post("/documents/ingest", response_model=IngestionResponse)
def ingest_documents() -> IngestionResponse:
    provider: EmbeddingProvider | None = None
    try:
        chunks = load_document_chunks()
        chunks_by_document = group_chunks_by_document(chunks)
        provider = GeminiEmbeddingProvider()
        embeddings = provider.embed_documents([chunk.content for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise RuntimeError("A quantidade de embeddings não corresponde aos trechos.")

        embeddings_by_document: dict[str, list[list[float]]] = {}
        offset = 0
        for document_id, document_chunks in chunks_by_document.items():
            next_offset = offset + len(document_chunks)
            embeddings_by_document[document_id] = embeddings[offset:next_offset]
            offset = next_offset

        store = ChromaVectorStore()
        stored_count = store.replace_documents(chunks_by_document, embeddings_by_document)
        return IngestionResponse(
            documents_processed=len(chunks_by_document),
            chunks_stored=stored_count,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Falha ao ingerir documentos: {error}") from error
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            close()


@app.get("/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    try:
        documents = ChromaVectorStore().list_documents()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Falha ao consultar documentos: {error}") from error
    return DocumentsResponse(documents=documents, total=len(documents))


@app.delete("/documents/{document_id:path}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str) -> DeleteDocumentResponse:
    try:
        deleted_count = ChromaVectorStore().delete_document(document_id)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Falha ao remover documento: {error}") from error

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Documento não encontrado na base vetorial.")
    return DeleteDocumentResponse(document_id=document_id, chunks_deleted=deleted_count)
