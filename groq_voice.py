"""Groq API helpers for transcription, structured order parsing, and speech."""
from __future__ import annotations

import io
import json
import os
import re
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
def load_local_env() -> None:
    """Read simple KEY=VALUE entries without replacing exported environment."""
    env_file = ROOT / ".env"
    if not env_file.exists(): return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        name, value = line.split("=", 1)
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'": value = value[1:-1]
        os.environ.setdefault(name.strip(), value)

load_local_env()
STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "openai/gpt-oss-120b"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"
TTS_LIMIT = 200

def client() -> Any:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to the project .env file and restart the app.")
    try:
        from groq import AsyncGroq
    except ImportError as exc:
        raise RuntimeError("Groq SDK is not installed. Run python -m pip install -r requirements.txt.") from exc
    return AsyncGroq(api_key=key)

def pcm16_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate); wav.writeframes(pcm)
    return out.getvalue()

def split_tts_text(text: str, limit: int = TTS_LIMIT) -> list[str]:
    words = text.split()
    chunks: list[str] = []; current = ""
    for word in words:
        if len(word) > limit:
            raise ValueError("A word in the reply exceeds Groq Orpheus's 200 character input limit.")
        candidate = f"{current} {word}".strip()
        if len(candidate) <= limit:
            current = candidate
        else:
            if current: chunks.append(current)
            current = word
    if current: chunks.append(current)
    return chunks

def assemble_wav(parts: list[bytes]) -> bytes:
    if not parts: return pcm16_wav(b"")
    params = None; frames = []
    for data in parts:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getcomptype() != "NONE": raise ValueError("Groq TTS returned a compressed WAV stream.")
            if params is None: params = wav.getparams()
            elif (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (
                params.nchannels, params.sampwidth, params.framerate, params.comptype
            ): raise ValueError("Groq TTS WAV chunks have incompatible formats.")
            frames.append(wav.readframes(wav.getnframes()))
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setparams(params); wav.writeframes(b"".join(frames))
    return out.getvalue()

async def transcribe(pcm: bytes) -> str:
    api = client()
    try:
        response = await api.audio.transcriptions.create(file=("speech.wav", pcm16_wav(pcm), "audio/wav"), model=STT_MODEL, language="en", response_format="json")
        return (response.text or "").strip()
    except Exception as exc:
        raise RuntimeError(f"Groq transcription failed: {friendly_error(exc)}") from exc
    finally: await api.close()

def friendly_error(exc: Exception) -> str:
    message = str(exc)
    if "model_terms_required" in message:
        return "Groq Orpheus model terms must be accepted by an organization admin in Groq Console before speech can be generated."
    if "401" in message or "invalid_api_key" in message.lower(): return "Groq rejected the API key. Check GROQ_API_KEY in .env."
    if getattr(exc, "status_code", None) == 429 or "429" in message:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after")
        wait = f" Retry after {retry_after} seconds." if retry_after else ""
        return (
            "Groq rate limit reached."
            f"{wait} Check this model's rate limits and usage in Groq Console; "
            "limits can apply per minute or per day."
        )
    return message[:500]

async def parse_order(user_text: str, history: list[dict[str, str]], menu: list[dict[str, Any]]) -> dict[str, Any]:
    api = client()
    schema = {"type":"object","properties":{"reply":{"type":"string"},"tool":{"type":["string","null"],"enum":["show_menu","add_to_cart","remove_from_cart","remove_item_from_cart","set_item_quantity","replace_cart_item","get_cart",None]},"arguments":{"type":"string"}},"required":["reply","tool","arguments"],"additionalProperties":False}
    messages = [{"role":"system","content":"You are a concise English restaurant ordering assistant. Use only the provided menu. For menu/cart actions select the matching tool and put its function arguments as a JSON encoded string in arguments; never invent menu items or prices. Ask clarification if needed. A tool result must inform your reply. This is a mock ordering demo."},{"role":"system","content":"ACTIVE MENU JSON: "+json.dumps(menu, ensure_ascii=False)}]
    messages.extend(history[-10:]); messages.append({"role":"user","content":user_text})
    try:
        result = await api.chat.completions.create(model=LLM_MODEL,messages=messages,response_format={"type":"json_schema","json_schema":{"name":"restaurant_turn","strict":True,"schema":schema}})
        parsed = json.loads(result.choices[0].message.content or "{}")
        if isinstance(parsed.get("arguments"), str):
            try: parsed["arguments"] = json.loads(parsed["arguments"] or "{}")
            except json.JSONDecodeError: parsed["arguments"] = {}
        return parsed
    except Exception as exc: raise RuntimeError(f"Groq order parsing failed: {friendly_error(exc)}") from exc
    finally: await api.close()

async def synthesize(text: str) -> bytes:
    api=client(); parts=[]
    try:
        for chunk in split_tts_text(text):
            response=await api.audio.speech.create(model=TTS_MODEL,voice=TTS_VOICE,input=chunk,response_format="wav")
            parts.append(await response.read())
        return assemble_wav(parts)
    except Exception as exc: raise RuntimeError(f"Groq speech synthesis failed: {friendly_error(exc)}") from exc
    finally: await api.close()
