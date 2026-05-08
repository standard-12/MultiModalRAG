"""
core/vector_store.py
--------------------
MultiModalVectorStore manages three ChromaDB persistent collections:

  text_docs   — text chunks          (all-MiniLM-L6-v2, 384-dim, cosine)
  image_docs  — image captions/CLIP  (clip-ViT-B-32,    512-dim, cosine)
  audio_docs  — audio transcripts    (all-MiniLM-L6-v2, 384-dim, cosine)

Public interface is intentionally small:
  add_text / add_image / add_audio /
  search_text / search_images / search_audio /
  get_stats / is_empty
"""

import uuid
from typing import Any, Dict, List

import chromadb

from .embedding_models import ClipEmbeddingModel, TextEmbeddingModel


class MultiModalVectorStore:
    """Persistent ChromaDB-backed vector store for text and image modalities."""

    def __init__(self, persist_dir: str = "./chroma_db") -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)

        self._text_embedder = TextEmbeddingModel()
        self._clip_embedder = ClipEmbeddingModel()

        self.text_collection = self._client.get_or_create_collection(
            name="text_docs",
            embedding_function=self._text_embedder,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_collection = self._client.get_or_create_collection(
            name="image_docs",
            embedding_function=self._clip_embedder,
            metadata={"hnsw:space": "cosine"},
        )
        # Audio transcripts share the same text embedding space
        self.audio_collection = self._client.get_or_create_collection(
            name="audio_docs",
            embedding_function=self._text_embedder,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def clip_model(self):
        """Raw SentenceTransformer CLIP model for external callers (e.g. ImageIngestionService)."""
        return self._clip_embedder.raw_model

    # ── Write operations ──────────────────────────────────────────────────────

    def add_text(self, text: str, metadata: Dict[str, Any]) -> str:
        """Index a text chunk and return its generated document ID."""
        doc_id = str(uuid.uuid4())
        # ChromaDB requires metadata values to be scalar types
        safe_meta = {k: str(v) for k, v in metadata.items()}
        self.text_collection.add(ids=[doc_id], documents=[text], metadatas=[safe_meta])
        return doc_id

    def add_image(
        self,
        caption: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> str:
        """Index an image by its caption + pre-computed CLIP embedding."""
        doc_id = str(uuid.uuid4())
        safe_meta = {k: str(v) for k, v in metadata.items()}
        self.image_collection.add(
            ids=[doc_id],
            documents=[caption],
            embeddings=[embedding],
            metadatas=[safe_meta],
        )
        return doc_id

    def add_audio(self, chunk_text: str, metadata: Dict[str, Any]) -> str:
        """Index an audio transcript chunk using the text embedding model."""
        doc_id    = str(uuid.uuid4())
        safe_meta = {k: str(v) for k, v in metadata.items()}
        self.audio_collection.add(
            ids=[doc_id],
            documents=[chunk_text],
            metadatas=[safe_meta],
        )
        return doc_id

    # ── Read / search operations ──────────────────────────────────────────────

    def search_text(self, query: str, n_results: int = 6) -> List[Dict[str, Any]]:
        """Semantic search over text chunks; returns up to *n_results* hits."""
        n = min(n_results, self.text_collection.count())
        if n == 0:
            return []
        raw = self.text_collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(raw, modality="text")

    def search_images(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Cross-modal image search using CLIP text-query embedding."""
        n = min(n_results, self.image_collection.count())
        if n == 0:
            return []
        query_embedding = self._clip_embedder.encode_text(query)
        raw = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(raw, modality="image")

    def search_audio(self, query: str, n_results: int = 4) -> List[Dict[str, Any]]:
        """Semantic search over audio transcript chunks."""
        n = min(n_results, self.audio_collection.count())
        if n == 0:
            return []
        raw = self.audio_collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(raw, modality="audio")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        text_count  = self.text_collection.count()
        image_count = self.image_collection.count()
        audio_count = self.audio_collection.count()
        return {
            "text_chunks":   text_count,
            "images":        image_count,
            "audio_chunks":  audio_count,
            "total":         text_count + image_count + audio_count,
        }

    def is_empty(self) -> bool:
        return self.get_stats()["total"] == 0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _format_results(self, raw: Dict, modality: str) -> List[Dict[str, Any]]:
        """Convert raw ChromaDB response into a normalised list of result dicts."""
        if not raw["ids"] or not raw["ids"][0]:
            return []
        results = []
        for i, doc_id in enumerate(raw["ids"][0]):
            distance = raw["distances"][0][i]
            results.append(
                {
                    "id":       doc_id,
                    "text":     raw["documents"][0][i],
                    "metadata": raw["metadatas"][0][i],
                    "score":    max(0.0, 1.0 - distance),  # cosine similarity
                    "modality": modality,
                    "rank":     i + 1,
                }
            )
        return results
