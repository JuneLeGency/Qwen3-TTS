#!/usr/bin/env python3
"""Qwen3-TTS FastAPI server — 3 models (CustomVoice, VoiceDesign, Base/Clone).

Models at startup: CustomVoice (0.6B) + Base (0.6B) — always loaded.
VoiceDesign (1.7B) — lazy-loaded on first use to save VRAM.

All endpoints support automatic text chunking for long texts.
"""

import base64
import io
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qwen_tts import Qwen3TTSModel

logger = logging.getLogger("tts-server")

# --- Model paths ---
MODEL_CUSTOM_VOICE = "./Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_VOICE_DESIGN = "./Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_BASE_CLONE = "./Qwen3-TTS-12Hz-0.6B-Base"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

# --- Models ---
model_custom_voice: Optional[Qwen3TTSModel] = None
model_base_clone: Optional[Qwen3TTSModel] = None
model_voice_design: Optional[Qwen3TTSModel] = None
_voice_design_lock = threading.Lock()

# --- Chunking config ---
MAX_CHUNK_CHARS = 200
SILENCE_MS = 300

LOAD_KWARGS = dict(dtype=DTYPE, attn_implementation="flash_attention_2")


def _get_voice_design_model() -> Qwen3TTSModel:
    """Lazy-load VoiceDesign model on first use."""
    global model_voice_design
    if model_voice_design is not None:
        return model_voice_design
    with _voice_design_lock:
        if model_voice_design is not None:
            return model_voice_design
        logger.info("Lazy-loading VoiceDesign (1.7B)...")
        t0 = time.time()
        model_voice_design = Qwen3TTSModel.from_pretrained(
            MODEL_VOICE_DESIGN, device_map=DEVICE, **LOAD_KWARGS,
        )
        logger.info("VoiceDesign loaded in %.1fs", time.time() - t0)
        return model_voice_design


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text at sentence boundaries, each chunk <= max_chars."""
    sentences = re.split(r"(?<=[。！？.!?\n])", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sent), max_chars):
                chunks.append(sent[i : i + max_chars])
            continue
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current)
            current = sent
        else:
            current += sent
    if current:
        chunks.append(current)
    return chunks


def concat_audio(arrays: list[np.ndarray], sr: int, silence_ms: int = SILENCE_MS) -> np.ndarray:
    """Concatenate audio arrays with silence padding between them."""
    if len(arrays) == 1:
        return arrays[0]
    silence = np.zeros(int(sr * silence_ms / 1000), dtype=arrays[0].dtype)
    parts: list[np.ndarray] = []
    for i, arr in enumerate(arrays):
        parts.append(arr)
        if i < len(arrays) - 1:
            parts.append(silence)
    return np.concatenate(parts)


def encode_audio(wav: np.ndarray, sr: int, fmt: str = "wav") -> str:
    """Encode audio array to base64 string."""
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format=fmt.upper())
    return base64.b64encode(buf.getvalue()).decode()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_custom_voice, model_base_clone

    logger.info("Loading CustomVoice (0.6B)...")
    t0 = time.time()
    model_custom_voice = Qwen3TTSModel.from_pretrained(
        MODEL_CUSTOM_VOICE, device_map=DEVICE, **LOAD_KWARGS,
    )
    logger.info("CustomVoice loaded in %.1fs", time.time() - t0)

    logger.info("Loading Base/Clone (0.6B)...")
    t0 = time.time()
    model_base_clone = Qwen3TTSModel.from_pretrained(
        MODEL_BASE_CLONE, device_map=DEVICE, **LOAD_KWARGS,
    )
    logger.info("Base/Clone loaded in %.1fs", time.time() - t0)

    logger.info("VoiceDesign (1.7B) will be lazy-loaded on first use.")
    yield
    model_custom_voice = model_base_clone = None


app = FastAPI(title="Qwen3-TTS API (3 Models)", lifespan=lifespan)


# ─── Request/Response Models ────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    speaker: str = Field(default="Vivian", description="Speaker name")
    language: str = Field(default="Auto", description="Language")
    instruct: Optional[str] = Field(default=None, description="Voice style instruction")
    format: str = Field(default="wav", description="Audio format: wav or mp3")


class VoiceDesignRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    instruct: str = Field(..., description="Natural language voice description")
    language: str = Field(default="Auto", description="Language")
    format: str = Field(default="wav", description="Audio format: wav or mp3")


class VoiceCloneRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    ref_audio_base64: str = Field(..., description="Base64-encoded reference audio (3-10s WAV)")
    ref_text: Optional[str] = Field(default=None, description="Transcript of reference audio")
    language: str = Field(default="Auto", description="Language")
    format: str = Field(default="wav", description="Audio format: wav or mp3")


class AudioResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    format: str
    duration_seconds: float
    chunks: int = Field(default=1, description="Number of text chunks synthesized")


# ─── Health & Info ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "custom_voice": model_custom_voice is not None,
            "voice_design": model_voice_design is not None,
            "base_clone": model_base_clone is not None,
        },
    }


@app.get("/speakers")
async def speakers():
    if model_custom_voice is None:
        raise HTTPException(503, "CustomVoice model not loaded")
    spks = model_custom_voice.model.get_supported_speakers()
    langs = model_custom_voice.model.get_supported_languages()
    return {"speakers": list(spks) if spks else [], "languages": list(langs) if langs else []}


# ─── Endpoint 1: CustomVoice TTS ───────────────────────────────────

@app.post("/v1/tts", response_model=AudioResponse)
async def synthesize_custom_voice(req: TTSRequest):
    """Generate speech with preset speakers + optional style instructions."""
    if model_custom_voice is None:
        raise HTTPException(503, "CustomVoice model not loaded")

    chunks = split_text(req.text)
    if not chunks:
        raise HTTPException(400, "Empty text")

    logger.info("[CustomVoice] %d chunk(s), %d chars, speaker=%s",
                len(chunks), len(req.text), req.speaker)

    audio_arrays: list[np.ndarray] = []
    sample_rate = None

    for i, chunk in enumerate(chunks):
        logger.info("  Chunk %d/%d (%d chars): %s...",
                     i + 1, len(chunks), len(chunk), chunk[:40])
        try:
            wavs, sr = model_custom_voice.generate_custom_voice(
                text=chunk, speaker=req.speaker,
                language=req.language, instruct=req.instruct,
            )
        except Exception as e:
            raise HTTPException(500, f"CustomVoice failed on chunk {i + 1}: {e}")
        audio_arrays.append(wavs[0])
        sample_rate = sr

    full_wav = concat_audio(audio_arrays, sample_rate)
    return AudioResponse(
        audio_base64=encode_audio(full_wav, sample_rate, req.format),
        sample_rate=sample_rate,
        format=req.format,
        duration_seconds=round(len(full_wav) / sample_rate, 2),
        chunks=len(chunks),
    )


# ─── Endpoint 2: VoiceDesign (lazy-loaded) ─────────────────────────

@app.post("/v1/voice-design", response_model=AudioResponse)
async def synthesize_voice_design(req: VoiceDesignRequest):
    """Generate speech using a natural-language voice description.
    
    The VoiceDesign model (1.7B) is lazy-loaded on first use.
    """
    model = _get_voice_design_model()

    chunks = split_text(req.text)
    if not chunks:
        raise HTTPException(400, "Empty text")

    logger.info("[VoiceDesign] %d chunk(s), %d chars, instruct='%s...'",
                len(chunks), len(req.text), req.instruct[:50])

    audio_arrays: list[np.ndarray] = []
    sample_rate = None

    for i, chunk in enumerate(chunks):
        logger.info("  Chunk %d/%d (%d chars): %s...",
                     i + 1, len(chunks), len(chunk), chunk[:40])
        try:
            wavs, sr = model.generate_voice_design(
                text=chunk, instruct=req.instruct, language=req.language,
            )
        except Exception as e:
            raise HTTPException(500, f"VoiceDesign failed on chunk {i + 1}: {e}")
        audio_arrays.append(wavs[0])
        sample_rate = sr

    full_wav = concat_audio(audio_arrays, sample_rate)
    return AudioResponse(
        audio_base64=encode_audio(full_wav, sample_rate, req.format),
        sample_rate=sample_rate,
        format=req.format,
        duration_seconds=round(len(full_wav) / sample_rate, 2),
        chunks=len(chunks),
    )


# ─── Endpoint 3: Voice Clone (Base) ────────────────────────────────

@app.post("/v1/voice-clone", response_model=AudioResponse)
async def synthesize_voice_clone(req: VoiceCloneRequest):
    """Clone a voice from reference audio using the Base model."""
    if model_base_clone is None:
        raise HTTPException(503, "Base/Clone model not loaded")

    chunks = split_text(req.text)
    if not chunks:
        raise HTTPException(400, "Empty text")

    try:
        ref_bytes = base64.b64decode(req.ref_audio_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 reference audio")

    logger.info("[VoiceClone] %d chunk(s), %d chars, ref_text=%s",
                len(chunks), len(req.text),
                f"'{req.ref_text[:30]}...'" if req.ref_text else "None")

    ref_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(ref_bytes)
            ref_path = f.name

        audio_arrays: list[np.ndarray] = []
        sample_rate = None

        for i, chunk in enumerate(chunks):
            logger.info("  Chunk %d/%d (%d chars): %s...",
                         i + 1, len(chunks), len(chunk), chunk[:40])
            try:
                wavs, sr = model_base_clone.generate_voice_clone(
                    text=chunk, ref_audio=ref_path,
                    ref_text=req.ref_text, language=req.language,
                    x_vector_only_mode=not bool(req.ref_text),
                )
            except Exception as e:
                raise HTTPException(500, f"VoiceClone failed on chunk {i + 1}: {e}")
            audio_arrays.append(wavs[0])
            sample_rate = sr
    finally:
        if ref_path:
            try:
                os.unlink(ref_path)
            except OSError:
                pass

    full_wav = concat_audio(audio_arrays, sample_rate)
    return AudioResponse(
        audio_base64=encode_audio(full_wav, sample_rate, req.format),
        sample_rate=sample_rate,
        format=req.format,
        duration_seconds=round(len(full_wav) / sample_rate, 2),
        chunks=len(chunks),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=9092)
