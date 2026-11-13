import hashlib
from pathlib import Path

from pydantic import BaseModel, Field


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
SUPPORTED_EXTENSIONS = {".md", ".txt"}


class KnowledgeDocument(BaseModel):
    document_id: str
    source: str
    category: str
    content: str


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    category: str
    chunk_index: int = Field(ge=0)
    content: str


def load_knowledge_documents(
    directory: str | Path = KNOWLEDGE_BASE_DIR,
) -> list[KnowledgeDocument]:
    """Read TXT and Markdown documents and attach stable source metadata."""
    knowledge_base = Path(directory)
    if not knowledge_base.is_dir():
        raise FileNotFoundError(f"Pasta da base de conhecimento não encontrada: {knowledge_base}")

    documents: list[KnowledgeDocument] = []
    files = sorted(
        (
            path
            for path in knowledge_base.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )

    for path in files:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = path.relative_to(knowledge_base).as_posix()
        parent_category = path.relative_to(knowledge_base).parent.as_posix()
        category = "geral" if parent_category == "." else parent_category

        documents.append(
            KnowledgeDocument(
                document_id=relative_path,
                source=relative_path,
                category=category,
                content=content,
            )
        )

    return documents


def split_document(
    document: KnowledgeDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split a document into character windows and create deterministic IDs."""
    if chunk_size <= 0:
        raise ValueError("chunk_size precisa ser maior que zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap precisa ser maior ou igual a zero e menor que chunk_size.")

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < len(document.content):
        end = min(start + chunk_size, len(document.content))
        chunk_content = document.content[start:end].strip()

        if chunk_content:
            content_hash = hashlib.sha256(chunk_content.encode("utf-8")).hexdigest()
            id_input = f"{document.document_id}:{chunk_index}:{content_hash}"
            chunk_id = hashlib.sha256(id_input.encode("utf-8")).hexdigest()

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source=document.source,
                    category=document.category,
                    chunk_index=chunk_index,
                    content=chunk_content,
                )
            )

        if end == len(document.content):
            break

        start = end - overlap
        chunk_index += 1

    return chunks


def load_document_chunks(
    directory: str | Path = KNOWLEDGE_BASE_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Read all supported documents in a folder and split them into chunks."""
    return [
        chunk
        for document in load_knowledge_documents(directory)
        for chunk in split_document(document, chunk_size=chunk_size, overlap=overlap)
    ]
