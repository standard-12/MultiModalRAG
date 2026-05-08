@echo off
setlocal enabledelayedexpansion

echo ============================================
echo    NexusRAG - Setup
echo ============================================
echo.

:: ── Colors (limited support in cmd) ───────────
set GREEN=[OK]
set WARN=[!!]
set ERR=[ERR]
set INFO=[--]

:: ── 1. Check Python ───────────────────────────
echo 1. Checking Python...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo %ERR% Python not found. Install Python 3.9+
    exit /b 1
)

for /f "tokens=2 delims= " %%i in ('python --version') do set PY_VER=%%i
echo %GREEN% Python !PY_VER! found

:: ── 2. Virtual environment ────────────────────
echo.
echo 2. Setting up virtual environment...

if not exist venv (
    python -m venv venv
    echo %GREEN% Virtual environment created
) else (
    echo %GREEN% Virtual environment already exists
)

call venv\Scripts\activate
echo %GREEN% Virtual environment activated

:: ── 3. Upgrade pip ────────────────────────────
echo.
echo 3. Upgrading pip...
python -m pip install --upgrade pip >nul 2>nul
echo %GREEN% pip upgraded

:: ── 4. Install UV (optional) ──────────────────
echo.
echo 4. Installing UV (optional)...

pip install -q uv >nul 2>nul
if %errorlevel% neq 0 (
    echo %WARN% UV install failed, will use pip
    set USE_UV=0
) else (
    echo %GREEN% UV installed
    set USE_UV=1
)

:: ── 5. Install dependencies ───────────────────
echo.
echo 5. Installing dependencies...
echo %INFO% This may take time...

if "!USE_UV!"=="1" (
    uv pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo %WARN% UV failed, falling back to pip
        pip install -r requirements.txt
    )
) else (
    pip install -r requirements.txt
)

echo %GREEN% Dependencies installed

:: ── 6. Verify imports ─────────────────────────
echo.
echo 6. Verifying key imports...

python -c "import chromadb; print('chromadb OK')" || echo %WARN% chromadb failed
python -c "import sentence_transformers; print('sentence-transformers OK')" || echo %WARN% sentence-transformers failed
python -c "import fitz; print('PyMuPDF OK')" || echo %WARN% PyMuPDF failed
python -c "import langchain; print('langchain OK')" || echo %WARN% langchain failed
python -c "import langgraph; print('langgraph OK')" || echo %WARN% langgraph failed
python -c "from PIL import Image; print('Pillow OK')" || echo %WARN% Pillow failed

:: ── 7. Ollama check ───────────────────────────
echo.
echo 7. Checking Ollama...

where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo %WARN% Ollama not found
    echo Install from: https://ollama.com
    echo Then run:
    echo   ollama pull qwen2:0.5b
    echo   ollama pull llava
) else (
    echo %GREEN% Ollama found

    curl -s http://localhost:11434/api/tags >nul 2>nul
    if %errorlevel% neq 0 (
        echo %WARN% Ollama server not running
        echo Run: ollama serve
    ) else (
        echo %GREEN% Ollama server running

        echo Pulling qwen2:0.5b...
        ollama pull qwen2:0.5b

        set /p PULL_LLAVA=Pull llava (vision model)? [y/N]: 
        if /i "!PULL_LLAVA!"=="y" (
            ollama pull llava
        )
    )
)

:: ── 8. Create directories ─────────────────────
echo.
echo 8. Creating directories...

mkdir docs 2>nul
mkdir images 2>nul
mkdir audio 2>nul
mkdir uploads 2>nul
mkdir chroma_db 2>nul
mkdir frontend\static\thumbnails 2>nul

echo %GREEN% Directories ready

:: ── 9. Smoke test ─────────────────────────────
echo.
echo 9. Running import smoke test...

python -c ^
"import sys; sys.path.insert(0, '.'); ^
from backend.core.vector_store import MultiModalVectorStore; ^
from backend.ingestion.document_ingestion import DocumentIngestionService; ^
from backend.ingestion.image_ingestion import ImageIngestionService; ^
print('All backend modules import cleanly.')"

if %errorlevel% neq 0 (
    echo %WARN% Smoke test failed
) else (
    echo %GREEN% Smoke test passed
)

:: ── Done ─────────────────────────────────────
echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo To start:
echo.
echo   venv\Scripts\activate
echo   python server.py
echo.
echo Then open: http://127.0.0.1:5353
echo.
pause