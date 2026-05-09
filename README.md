# MultiModalRAG

A powerful Retrieval-Augmented Generation (RAG) system that supports **multiple modalities**: text documents, images, and audio. Combines advanced embedding models, vector search, and an offline LLM for intelligent document understanding and querying.

## Features

✨ **Multimodal Input Support**
- 📄 **Documents**: PDF, Markdown, plain text
- 🖼️ **Images**: JPG, PNG, WebP, BMP with CLIP embeddings
- 🎵 **Audio**: MP3, WAV, M4A, OGG, FLAC with Whisper transcription

🧠 **Advanced Retrieval**
- Semantic search using sentence transformers
- Cross-encoder reranking for relevance optimization
- Hybrid retrieval combining document and image context

🤖 **Offline LLM Integration**
- Ollama-based language models
- No API calls required—runs fully locally
- Configurable model selection (default: qwen2:0.5b)

📦 **Vector Storage**
- Chroma vector database for persistent storage
- Automatic indexing of uploaded files
- Efficient similarity search

🌐 **Web Interface**
- Modern, responsive frontend
- Real-time chat with RAG pipeline
- Document and image management
- Status monitoring dashboard

## Prerequisites

- **Python 3.11+** (recommended: Homebrew installation)
- **Ollama** installed and running locally on `http://localhost:11434`
  - Download from [ollama.ai](https://ollama.ai)
  - Pull required model: `ollama pull qwen2:0.5b`

## Installation

### 1. Clone and Navigate to Repository

```bash
cd MultiModalRAG-main
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure Ollama is Running

```bash
# In a new terminal
ollama serve
```

Verify connectivity with:
```bash
curl http://localhost:11434/api/tags
```

## Running the Application

```bash
python3.11 server.py
```

The application will start at **http://127.0.0.1:5353**

### Startup Sequence

1. Creates required data directories (`docs/`, `images/`, `audio/`, `chroma_db/`)
2. Initializes the multimodal vector store with embedding models
3. Loads CLIP model for image embeddings
4. Sets up Whisper for audio transcription
5. Connects to Ollama LLM
6. Loads cross-encoder reranker
7. Auto-ingests files from `docs/` and `images/` if vector store is empty
8. Serves the web frontend

## Project Structure

```
MultiModalRAG-main/
├── server.py                      # Flask application entry point
├── requirements.txt               # Python dependencies
├── backend/
│   ├── api/
│   │   └── routes.py             # API endpoints
│   ├── config/
│   │   └── settings.py           # Configuration & environment variables
│   ├── core/
│   │   ├── embedding_models.py   # Embedding & CLIP models
│   │   └── vector_store.py       # Chroma vector store management
│   ├── ingestion/
│   │   ├── document_ingestion.py # PDF, Markdown, text processing
│   │   ├── image_ingestion.py    # Image ingestion with CLIP
│   │   └── audio_ingestion.py    # Audio transcription with Whisper
│   └── services/
│       └── retrieval_service.py  # RAG pipeline & LLM orchestration
├── frontend/
│   ├── templates/
│   │   └── index.html            # Main web interface
│   └── static/
│       ├── css/main.css          # Styling
│       ├── js/chat.js            # Frontend logic
│       └── thumbnails/           # Generated image thumbnails
├── docs/                         # Place document files here
├── images/                       # Place image files here
├── audio/                        # Place audio files here
└── chroma_db/                    # Vector store (auto-created)
```

## API Endpoints

### Core Endpoints

**`GET /`**
- Serves the web interface

**`GET /api/status`**
- Returns system status, LLM connectivity, and vector store statistics

**`GET /api/documents`**
- Lists all ingested documents, images, and audio files

**`POST /api/ingest`**
- Uploads and ingests new documents/images/audio files

**`POST /api/query`**
- Sends a query to the RAG pipeline
- Returns retrieved context and LLM response

**`POST /api/chat` (Streaming)**
- Real-time streaming chat interface
- Streams LLM responses as they're generated

## Configuration

All settings are managed in [backend/config/settings.py](backend/config/settings.py). Override values via environment variables:

```bash
# LLM Configuration
export OLLAMA_HOST="http://localhost:11434"
export LLM_MODEL="qwen2:0.5b"

# Then run the server
python3.11 server.py
```

### Default Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `qwen2:0.5b` | Language model to use |
| `CHUNK_SIZE` | 800 | Document chunk size for embedding |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |

## Supported File Types

### Documents
- `.pdf` – PDF files
- `.md` – Markdown files
- `.txt` – Plain text files

### Images
- `.jpg`, `.jpeg` – JPEG images
- `.png` – PNG images
- `.webp` – WebP images
- `.bmp` – Bitmap images

### Audio
- `.mp3` – MP3 audio
- `.wav` – WAV audio
- `.m4a` – M4A audio
- `.ogg` – OGG audio
- `.flac` – FLAC audio
- `.webm` – WebM audio
- `.mp4` – MP4 video (audio extracted)

## Usage

### 1. Add Documents

Place files in the following directories:
- Documents → `docs/`
- Images → `images/`
- Audio → `audio/`

Files will be automatically ingested when the server starts.

### 2. Query the System

Use the web interface at `http://127.0.0.1:5353`:
- Type your question in the chat box
- The RAG pipeline retrieves relevant content from all modalities
- The LLM generates a response based on retrieved context

### 3. Monitor System Status

Visit `/api/status` or check the dashboard for:
- Document count in vector store
- LLM connectivity status
- System health metrics

## Architecture Overview

```
User Query
    ↓
Query Embedding (sentence-transformers)
    ↓
Vector Search (Chroma)
    ↓
Retrieved Context (docs + images + audio)
    ↓
Cross-Encoder Reranking
    ↓
Top Results → LLM Prompt
    ↓
Ollama LLM Generation
    ↓
Response to User
```

## Troubleshooting

### Ollama Connection Error
```
Error: Could not connect to Ollama at http://localhost:11434
```
**Solution**: Ensure Ollama is running in another terminal with `ollama serve`

### Model Not Found
```
Error: Model 'qwen2:0.5b' not found
```
**Solution**: Pull the model with `ollama pull qwen2:0.5b`

### Vector Store Empty
- Place files in `docs/` and `images/` directories
- Restart the server or manually trigger ingestion via the API

### Python Import Errors
```
ModuleNotFoundError: No module named 'bs4' or similar
```
**Solution**: Ensure you're using Python 3.11 and all dependencies are installed:
```bash
/opt/homebrew/bin/python3.11 -m pip install -r requirements.txt
```

## Performance Notes

- Initial startup may take 30-60 seconds while models load
- Image and audio processing are computationally intensive
- Large document sets benefit from running on machines with GPU support (currently CPU-optimized)

## Future Enhancements

- [ ] GPU acceleration support
- [ ] Web interface for file management
- [ ] Fine-tuning support for custom models
- [ ] Multi-language support
- [ ] Advanced filtering and search operators

## License

This project is provided as-is for research and development purposes.

## Support

For issues or questions, check the following:
1. Ensure Ollama is running and accessible
2. Verify all dependencies are installed (`pip list | grep langchain`)
3. Check system logs in the terminal for error messages
4. Verify file permissions on `docs/`, `images/`, and `audio/` directories
