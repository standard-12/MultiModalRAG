"""
ingestion/audio_ingestion.py
-----------------------------
Processes audio files for multimodal RAG ingestion:

  1. Transcribes audio to text using OpenAI Whisper (runs fully offline).
  2. Extracts basic waveform metadata (duration, sample-rate) via soundfile.
  3. Chunks the transcript into overlapping text segments so long recordings
     are retrieved at the right granularity.
  4. Returns one dict per chunk, ready for insertion into the vector store.

Supported formats: MP3, WAV, M4A, OGG, FLAC, WEBM.

Whisper model sizes (speed ↔ accuracy trade-off):
  tiny   – fastest,  lowest accuracy  (~1 GB VRAM)
  base   – default,  good accuracy    (~1 GB VRAM)
  small  – better,   ~2 GB VRAM
  medium – best OSS, ~5 GB VRAM

Set the WHISPER_MODEL environment variable to override.
"""

import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

# soundfile is optional — used only for waveform metadata
try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

try:
    import whisper as _whisper
    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False


class AudioIngestionService:
    """
    Converts raw audio files into transcribed, chunked records ready for
    insertion into the vector store.
    """

    SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}

    def __init__(
        self,
        whisper_model: str = "base",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self._model        = None          # loaded lazily on first use
        self._model_name   = os.getenv("WHISPER_MODEL", whisper_model)

        if not _HAS_WHISPER:
            print(
                "  [AudioIngestionService] openai-whisper not installed. "
                "Run: pip install openai-whisper"
            )
        else:
            print(f"  [AudioIngestionService] Whisper ready (model='{self._model_name}').")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_audio(
        self, file_path: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Transcribe and chunk a single audio file.

        Returns a list of chunk dicts (one per segment), each with:
            file_path, file_name, transcript (full), chunk_text, chunk_index,
            duration_s, sample_rate, total_chunks.

        Returns ``None`` if the extension is unsupported or transcription fails.
        """
        path = Path(file_path)
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return None

        # ── Waveform metadata (optional) ───────────────────────────────────
        duration_s   = 0.0
        sample_rate  = 0
        if _HAS_SF:
            try:
                info        = sf.info(file_path)
                duration_s  = round(info.duration, 2)
                sample_rate = info.samplerate
            except Exception:
                pass  # non-critical

        # ── Transcription ──────────────────────────────────────────────────
        transcript = self._transcribe(file_path)
        if transcript is None:
            return None

        # ── Chunk transcript ───────────────────────────────────────────────
        chunks = self._chunk_text(transcript)
        if not chunks:
            chunks = [transcript]  # use full transcript as single chunk

        total = len(chunks)
        base_meta = {
            "file_path":    str(file_path),
            "file_name":    path.name,
            "content_type": "audio",
            "transcript":   transcript[:2000],   # store first 2 k chars
            "duration_s":   str(duration_s),
            "sample_rate":  str(sample_rate),
            "total_chunks": str(total),
        }

        records = []
        for idx, chunk in enumerate(chunks):
            records.append(
                {
                    **base_meta,
                    "chunk_text":  chunk,
                    "chunk_index": str(idx),
                    "title":       f"{path.stem} [audio:{idx+1}/{total}]",
                }
            )

        print(
            f"  [AudioIngestionService] '{path.name}' → "
            f"{total} chunk(s), {duration_s}s"
        )
        return records

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self):
        """Lazily load the Whisper model on first transcription."""
        if self._model is None:
            if not _HAS_WHISPER:
                raise RuntimeError("openai-whisper is not installed.")
            print(f"  [AudioIngestionService] Loading Whisper '{self._model_name}' model…")
            self._model = _whisper.load_model(self._model_name)
        return self._model

    def _transcribe(self, file_path: str) -> Optional[str]:
        """Run Whisper transcription; return the full transcript string."""
        try:
            model  = self._load_model()
            result = model.transcribe(file_path, fp16=False, verbose=False)
            text   = result.get("text", "").strip()
            return text if text else None
        except Exception as exc:
            print(f"  [AudioIngestionService] Transcription failed for {file_path}: {exc}")
            return None

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split *text* into overlapping character-level chunks.
        Breaks at word boundaries to avoid mid-word splits.
        """
        words  = text.split()
        chunks: List[str] = []
        buf    = []
        length = 0

        for word in words:
            wl = len(word) + 1  # +1 for space
            if length + wl > self.chunk_size and buf:
                chunks.append(" ".join(buf))
                # Overlap: keep last N characters worth of words
                overlap_buf: List[str] = []
                overlap_len = 0
                for w in reversed(buf):
                    if overlap_len + len(w) + 1 > self.chunk_overlap:
                        break
                    overlap_buf.insert(0, w)
                    overlap_len += len(w) + 1
                buf    = overlap_buf
                length = overlap_len
            buf.append(word)
            length += wl

        if buf:
            chunks.append(" ".join(buf))

        return chunks
