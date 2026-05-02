"""
server.py
---------
Flask application entry point — Advanced Multimodal RAG
Runs on: http://127.0.0.1:5353

Startup sequence
----------------
1. Ensure all data directories exist.
2. Lazily initialise the service container on first request.
3. Auto-ingest docs/ and images/ if the vector store is empty.
4. Register API routes.
5. Serve the frontend from frontend/templates and frontend/static.
"""

import sys
from pathlib import Path

from flask import Flask, render_template

# Make the repo root importable regardless of working directory
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from backend.api.routes import register_routes
from backend.config.settings import (
    CHROMA_DIR,
    DOCS_DIR,
    IMAGES_DIR,
    LLM_MODEL,
    OLLAMA_HOST,
    SUPPORTED_IMG_EXTENSIONS,
    THUMB_DIR,
    ensure_directories,
)
from backend.core.vector_store import MultiModalVectorStore
from backend.ingestion.document_ingestion import DocumentIngestionService
from backend.ingestion.image_ingestion import ImageIngestionService
from backend.services.retrieval_service import RetrievalService


# ── Flask application ─────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=str(ROOT_DIR / "frontend" / "templates"),
    static_folder=str(ROOT_DIR / "frontend" / "static"),
)



# ── Service container ─────────────────────────────────────────────────────────

class AppContainer:
    """Holds lazily-initialised singleton service instances."""

    def __init__(self):
        self.vector_store:      MultiModalVectorStore   | None = None
        self.doc_ingestion:     DocumentIngestionService | None = None
        self.img_ingestion:     ImageIngestionService   | None = None
        self.retrieval_service: RetrievalService        | None = None

    def initialise(self):
        """Boot all services; safe to call multiple times (no-op after first)."""
        if self.vector_store is not None:
            return

        ensure_directories()

        print("\n[startup] Loading MultiModal Vector Store …")
        self.vector_store = MultiModalVectorStore(persist_dir=str(CHROMA_DIR))

        print("[startup] Loading Document Ingestion Service …")
        self.doc_ingestion = DocumentIngestionService()

        print("[startup] Loading Image Ingestion Service …")
        self.img_ingestion = ImageIngestionService(
            ollama_host=OLLAMA_HOST,
            thumbnail_dir=str(THUMB_DIR),
        )

        print("[startup] Building Retrieval Service (RAG pipeline) …")
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            ollama_host=OLLAMA_HOST,
            model=LLM_MODEL,
        )

        # Auto-ingest on first run if knowledge base is empty
        if self.vector_store.is_empty():
            print("[startup] Knowledge base empty — ingesting docs/ and images/ …")
            doc_stats = self.doc_ingestion.ingest_directory(
                str(DOCS_DIR), self.vector_store
            )
            print(f"[startup] Text ingestion: {doc_stats}")
            img_count = self._ingest_image_directory(str(IMAGES_DIR))
            print(f"[startup] Images ingested: {img_count}")

        print("[startup] Ready.\n")

    def _ingest_image_directory(self, directory: str) -> int:
        """Walk *directory* and ingest all supported image files."""
        count = 0
        for fp in Path(directory).rglob("*"):
            if fp.suffix.lower() in SUPPORTED_IMG_EXTENSIONS:
                result = self.img_ingestion.process_image(
                    str(fp), clip_model=self.vector_store.clip_model
                )
                if result and result["embedding"]:
                    self.vector_store.add_image(
                        caption=result["caption"],
                        embedding=result["embedding"],
                        metadata={
                            "file_path":    str(fp),
                            "file_name":    result["file_name"],
                            "thumbnail":    result["thumbnail"],
                            "content_type": "image",
                        },
                    )
                    count += 1
        return count


_container = AppContainer()


# ── Middleware: lazy init on every request ────────────────────────────────────

@app.before_request
def ensure_initialised():
    _container.initialise()


# ── Frontend route ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")



# ── Wire up API routes ────────────────────────────────────────────────────────

register_routes(app, _container)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _container.initialise()
    app.run(host="127.0.0.1", port=5353, debug=False)
