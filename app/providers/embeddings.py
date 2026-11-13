import os
from pathlib import Path
from typing import Literal, Protocol, Sequence

from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072


class EmbeddingProvider(Protocol):
    """Common interface for document and question embedding providers."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector for each document text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return one embedding vector for a search query."""
        ...


class GeminiEmbeddingProvider:
    """Create Portuguese-capable embeddings through the Gemini Developer API."""

    def __init__(self, client: genai.Client | None = None) -> None:
        load_dotenv(PROJECT_ROOT / ".env")

        if client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY não definida. Copie .env.example para .env "
                    "e configure a chave localmente."
                )
            client = genai.Client(api_key=api_key)

        self._client = client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        document_texts = list(texts)
        if not document_texts:
            return []
        if any(not text.strip() for text in document_texts):
            raise ValueError("Os textos dos documentos não podem estar vazios.")

        return self._embed(
            document_texts,
            task_type="RETRIEVAL_DOCUMENT",
        )

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("A pergunta não pode estar vazia.")

        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def close(self) -> None:
        """Close the SDK client and its HTTP resources."""
        self._client.close()

    def _embed(
        self,
        texts: list[str],
        task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"],
    ) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

        if response.embeddings is None or len(response.embeddings) != len(texts):
            raise RuntimeError("O Gemini não retornou um embedding para cada texto.")

        vectors: list[list[float]] = []
        for embedding in response.embeddings:
            if embedding.values is None:
                raise RuntimeError("O Gemini retornou um embedding sem valores.")
            vectors.append([float(value) for value in embedding.values])

        return vectors
