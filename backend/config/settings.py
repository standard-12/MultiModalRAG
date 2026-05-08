"""
config/settings.py
------------------
Single source of truth for all environment-level constants.
Override values via environment variables where needed.
"""

import os
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
# Resolve relative to the repo root (two levels up from this file).
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent

DOCS_DIR:   Path = ROOT_DIR / "docs"
IMAGES_DIR: Path = ROOT_DIR / "images"
AUDIO_DIR:  Path = ROOT_DIR / "audio"
CHROMA_DIR: Path = ROOT_DIR / "chroma_db"
# Thumbnails are written inside the frontend static tree so Flask serves them
# automatically under /static/thumbnails/<filename>
THUMB_DIR:  Path = ROOT_DIR / "frontend" / "static" / "thumbnails"

# ── LLM / Ollama ──────────────────────────────────────────────────────────────
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL:   str = os.getenv("LLM_MODEL",   "qwen2:0.5b")

# ── Supported file types ──────────────────────────────────────────────────────
SUPPORTED_DOC_EXTENSIONS:   set[str] = {".pdf", ".md", ".txt"}
SUPPORTED_IMG_EXTENSIONS:   set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_AUDIO_EXTENSIONS: set[str] = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE:    int = 800
CHUNK_OVERLAP: int = 150


def ensure_directories() -> None:
    """Create all required data directories if they don't exist."""
    for directory in (DOCS_DIR, IMAGES_DIR, AUDIO_DIR, CHROMA_DIR, THUMB_DIR):
        directory.mkdir(parents=True, exist_ok=True)
