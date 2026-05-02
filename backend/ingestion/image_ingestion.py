"""
ingestion/image_ingestion.py
-----------------------------
Processes image files for multimodal RAG ingestion:

  1. Generates a natural-language caption via a LLaVA-family vision model
     (falls back to the filename if no vision model is available).
  2. Creates a 300×300 JPEG thumbnail for the chat UI.
  3. Computes a CLIP embedding using the provided SentenceTransformer model.

Supported formats: JPEG, PNG, WEBP, BMP, GIF.
"""

import base64
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from PIL import Image


class ImageIngestionService:
    """
    Converts raw image files into captioned, embedded records ready for
    insertion into the vector store.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        thumbnail_dir: str = "./static/thumbnails",
    ) -> None:
        self.ollama_host    = ollama_host
        self.thumbnail_dir  = Path(thumbnail_dir)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

        self._vision_model: Optional[str] = self._detect_vision_model()
        if self._vision_model:
            print(f"  Vision model available: {self._vision_model}")
        else:
            print("  No vision model found — image captions will use filename only.")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_image(
        self, file_path: str, clip_model=None
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single image file.

        Returns a dict with keys:
            file_path, file_name, caption, thumbnail (URL path),
            width, height, embedding (list[float] | None).

        Returns ``None`` if the file extension is unsupported or processing fails.
        """
        path = Path(file_path)
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return None

        try:
            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")
                width, height = img_rgb.size

                # Build and save thumbnail
                thumb_name = f"{path.stem}_thumb.jpg"
                thumb_path = self.thumbnail_dir / thumb_name
                thumb = img_rgb.copy()
                thumb.thumbnail((300, 300), Image.LANCZOS)
                thumb.save(str(thumb_path), "JPEG", quality=85)

                # CLIP embedding (operates on the full-size image)
                embedding: Optional[list] = None
                if clip_model is not None:
                    orig_copy = img_rgb.copy()
                    embedding = clip_model.encode(orig_copy, show_progress_bar=False).tolist()

            caption = self._generate_caption(file_path) or f"Image: {path.name}"

            return {
                "file_path":  str(file_path),
                "file_name":  path.name,
                "caption":    caption,
                "thumbnail":  f"/static/thumbnails/{thumb_name}",
                "width":      width,
                "height":     height,
                "embedding":  embedding,
            }

        except Exception as exc:
            print(f"  [ImageIngestionService] Failed to process {file_path}: {exc}")
            return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_vision_model(self) -> Optional[str]:
        """Poll Ollama for any available LLaVA-family vision model."""
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            if resp.status_code == 200:
                for model in resp.json().get("models", []):
                    name = model.get("name", "")
                    if any(v in name.lower() for v in ["llava", "bakllava", "moondream"]):
                        return name
        except Exception:
            pass
        return None

    def _generate_caption(self, file_path: str) -> Optional[str]:
        """Ask the vision model to describe the image; returns None on failure."""
        if not self._vision_model:
            return None
        try:
            with open(file_path, "rb") as fh:
                b64_image = base64.b64encode(fh.read()).decode()

            payload = {
                "model":  self._vision_model,
                "prompt": (
                    "Describe this image concisely but thoroughly. "
                    "Include the main subject, visible text, colours, and context."
                ),
                "images":  [b64_image],
                "stream":  False,
                "options": {"temperature": 0.1},
            }
            resp = requests.post(
                f"{self.ollama_host}/api/generate", json=payload, timeout=60
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as exc:
            print(f"  [ImageIngestionService] Caption generation failed: {exc}")
        return None
