from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "support_knowledge"


class ChromaVectorStore:
    """Persist document chunks and their externally generated embeddings."""

    def __init__(self, path: str | Path = CHROMA_DB_DIR) -> None:
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def list_documents(self) -> list[dict[str, Any]]:
        records = self._collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = {}

        for metadata in records["metadatas"] or []:
            if not metadata:
                continue
            document_id = str(metadata["document_id"])
            document = grouped.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "source": str(metadata["source"]),
                    "category": str(metadata["category"]),
                    "chunk_count": 0,
                },
            )
            document["chunk_count"] += 1

        return [grouped[key] for key in sorted(grouped)]

    def has_chunks(self) -> bool:
        return self._collection.count() > 0

    def replace_documents(self, chunks_by_document: dict[str, list[Any]], embeddings_by_document: dict[str, list[list[float]]]) -> int:
        existing_ids = {item["document_id"] for item in self.list_documents()}
        incoming_ids = set(chunks_by_document)

        # All embeddings are prepared before this method is called. Replace each
        # document's records so edited files cannot leave obsolete trailing chunks.
        for document_id in sorted(existing_ids | incoming_ids):
            self._collection.delete(where={"document_id": document_id})

        chunk_count = 0
        for document_id in sorted(incoming_ids):
            chunks = chunks_by_document[document_id]
            embeddings = embeddings_by_document[document_id]
            if not chunks:
                continue
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=embeddings,
                documents=[chunk.content for chunk in chunks],
                metadatas=[
                    {
                        "document_id": chunk.document_id,
                        "source": chunk.source,
                        "category": chunk.category,
                        "chunk_index": chunk.chunk_index,
                    }
                    for chunk in chunks
                ],
            )
            chunk_count += len(chunks)

        return chunk_count

    def delete_document(self, document_id: str) -> int:
        records = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        deleted_count = len(records["ids"])
        if deleted_count:
            self._collection.delete(where={"document_id": document_id})
        return deleted_count

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.has_chunks():
            return []

        filters = {"category": category} if category else None
        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        ids = response["ids"][0]
        documents = response["documents"][0] or []
        metadatas = response["metadatas"][0] or []
        distances = response["distances"][0] or []
        results: list[dict[str, Any]] = []

        for chunk_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            if metadata is None or content is None or distance is None:
                continue
            results.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": str(metadata["document_id"]),
                    "source": str(metadata["source"]),
                    "category": str(metadata["category"]),
                    "chunk_index": int(metadata["chunk_index"]),
                    "content": content,
                    "distance": float(distance),
                }
            )
        return results


def group_chunks_by_document(chunks: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)
    return dict(grouped)
