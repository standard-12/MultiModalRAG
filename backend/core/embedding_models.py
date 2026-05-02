"""
core/embedding_models.py
------------------------
Thin wrappers around SentenceTransformer that implement the
ChromaDB EmbeddingFunction protocol.

Models used
-----------
- TextEmbeddingModel  : all-MiniLM-L6-v2  (384-dim cosine space)
- ClipEmbeddingModel  : clip-ViT-B-32     (512-dim cosine space, multimodal)
"""

from typing import List

from chromadb import Documents, EmbeddingFunction, Embeddings
from PIL import Image
from sentence_transformers import SentenceTransformer


class TextEmbeddingModel(EmbeddingFunction):
    """Sentence-Transformer embedding function for plain text chunks."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        print(f"  Loading text embedding model: {model_name} ...")
        self._model = SentenceTransformer(model_name)

    # ChromaDB protocol
    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        return self._model.encode(list(input), show_progress_bar=False).tolist()

    def encode_text(self, text: str) -> List[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


class ClipEmbeddingModel(EmbeddingFunction):
    """CLIP embedding function — supports both text queries and image inputs."""

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        print(f"  Loading CLIP model: {model_name} ...")
        self._model = SentenceTransformer(model_name)

    # ChromaDB protocol
    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        return self._model.encode(list(input), show_progress_bar=False).tolist()

    def encode_text(self, text: str) -> List[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()

    def encode_image(self, image: Image.Image) -> List[float]:
        return self._model.encode(image, show_progress_bar=False).tolist()

    @property
    def raw_model(self) -> SentenceTransformer:
        """Expose the underlying SentenceTransformer for direct use."""
        return self._model
