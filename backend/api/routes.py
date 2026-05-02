"""
api/routes.py
--------------
All Flask route handlers (blueprinted or direct).
Routes delegate to services; no business logic lives here.
"""

import json
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename

# The app container injects these references at startup via _container.
# We access them through a module-level reference set by register_routes().
_container = None

api_bp = Blueprint("api", __name__, url_prefix="/api")


def register_routes(app, container):
    """
    Attach the API blueprint to *app* and wire up the shared service container.

    Parameters
    ----------
    app       : Flask application instance
    container : AppContainer with .vector_store, .doc_ingestion,
                .img_ingestion, .retrieval_service attributes
    """
    global _container
    _container = container
    app.register_blueprint(api_bp)


# ── /api/status ───────────────────────────────────────────────────────────────

@api_bp.route("/status")
def get_status():
    """Return LLM connectivity and vector-store statistics."""
    stats    = _container.vector_store.get_stats()
    llm_ok   = _container.retrieval_service.check_llm_connection()
    return jsonify(
        {
            "status":       "ready",
            "llm_connected": llm_ok,
            "model":         _container.retrieval_service.model_name,
            "ollama_host":   _container.retrieval_service.ollama_host,
            "stats":         stats,
        }
    )


# ── /api/documents ────────────────────────────────────────────────────────────

@api_bp.route("/documents")
def list_documents():
    """List all files currently present in the docs/ and images/ directories."""
    from ..config.settings import (
        DOCS_DIR,
        IMAGES_DIR,
        SUPPORTED_DOC_EXTENSIONS,
        SUPPORTED_IMG_EXTENSIONS,
    )

    docs = sorted(
        f.name
        for f in DOCS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_DOC_EXTENSIONS
    )
    images = sorted(
        f.name
        for f in IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMG_EXTENSIONS
    )
    return jsonify({"documents": docs, "images": images})


# ── /api/chat ─────────────────────────────────────────────────────────────────

@api_bp.route("/chat", methods=["POST"])
def chat():
    """Non-streaming chat endpoint — runs the full pipeline and returns JSON."""
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    result = _container.retrieval_service.run(query)
    return jsonify(result)


# ── /api/chat/stream ──────────────────────────────────────────────────────────

@api_bp.route("/chat/stream", methods=["POST"])
def chat_stream():
    """
    Streaming chat endpoint.

    Emits SSE events in order:
        pipeline_details → trace → token (×N) → sources → done
    """
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    def _event_stream():
        result = _container.retrieval_service.run(query)

        # 1. Pipeline internals (HyDE doc, expanded queries, RRF table)
        yield (
            f"data: {json.dumps({'type': 'pipeline_details', 'hyde_document': result['hyde_document'], 'expanded_queries': result['expanded_queries'], 'rrf_results': result['rrf_results']})}\n\n"
        )

        # 2. Pipeline trace steps
        yield f"data: {json.dumps({'type': 'trace', 'steps': result['pipeline_trace']})}\n\n"

        # 3. Stream answer tokens word-by-word
        for word in result["answer"].split():
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

        # 4. Sources
        yield f"data: {json.dumps({'type': 'sources', 'sources': result['sources']})}\n\n"

        # 5. Done sentinel
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(_event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /api/upload ───────────────────────────────────────────────────────────────

@api_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Accept a multipart file upload, ingest it into the vector store, and
    return a JSON summary of what was indexed.
    """
    from ..config.settings import (
        DOCS_DIR,
        IMAGES_DIR,
        SUPPORTED_DOC_EXTENSIONS,
        SUPPORTED_IMG_EXTENSIONS,
    )

    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    ext      = Path(filename).suffix.lower()

    # ── Document upload ───────────────────────────────────────────────────────
    if ext in SUPPORTED_DOC_EXTENSIONS:
        save_path = DOCS_DIR / filename
        file.save(str(save_path))

        if ext == ".pdf":
            chunks = _container.doc_ingestion.process_pdf(str(save_path))
        else:
            chunks = _container.doc_ingestion.process_text_file(str(save_path))

        for chunk in chunks:
            _container.vector_store.add_text(chunk["text"], chunk["metadata"])

        return jsonify(
            {
                "message": f"Indexed {len(chunks)} chunks from {filename}",
                "type":    "document",
                "chunks":  len(chunks),
            }
        )

    # ── Image upload ──────────────────────────────────────────────────────────
    if ext in SUPPORTED_IMG_EXTENSIONS:
        save_path = IMAGES_DIR / filename
        file.save(str(save_path))

        result = _container.img_ingestion.process_image(
            str(save_path), clip_model=_container.vector_store.clip_model
        )
        if result and result["embedding"]:
            _container.vector_store.add_image(
                caption=result["caption"],
                embedding=result["embedding"],
                metadata={
                    "file_path":    str(save_path),
                    "file_name":    result["file_name"],
                    "thumbnail":    result["thumbnail"],
                    "content_type": "image",
                },
            )
            return jsonify(
                {
                    "message":   f"Indexed image {filename}",
                    "type":      "image",
                    "caption":   result["caption"],
                    "thumbnail": result["thumbnail"],
                }
            )
        return jsonify({"error": "Failed to process image"}), 500

    return jsonify({"error": f"Unsupported file type: {ext}"}), 400
