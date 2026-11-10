import asyncio
import io
import json
import os
import sys
import types
import unittest
import wave
from unittest.mock import patch


class FakeAudio:
    async def transcriptions_create(self, **kwargs):
        self.transcriptions_kwargs = kwargs
        return types.SimpleNamespace(text="two cokes")

    async def speech_create(self, **kwargs):
        self.speech_calls.append(kwargs)
        return types.SimpleNamespace(read=self.wav_chunk)

    async def wav_chunk(self):
        out = io.BytesIO()
        with wave.open(out, "wb") as f:
            frame_count = 20 + len(self.speech_calls)
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(24000); f.writeframes(b"\x01\x00" * frame_count)
        return out.getvalue()

    def __init__(self):
        self.speech_calls = []
        self.transcriptions = types.SimpleNamespace(create=self.transcriptions_create)
        self.speech = types.SimpleNamespace(create=self.speech_create)


class FakeGroq:
    instances = []

    def __init__(self, **kwargs):
        self.__class__.instances.append(self)
        self.api_key = kwargs.get("api_key")
        self.audio = FakeAudio()
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.chat_create))
        self.closed = False

    async def chat_create(self, **kwargs):
        self.chat_kwargs = kwargs
        payload = {"reply": "Added.", "tool": "add_to_cart", "arguments": json.dumps({"item_name": "Coke", "quantity": 2})}
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(payload)))])

    async def close(self): self.closed = True


class GroqVoiceMockSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake_module = types.ModuleType("groq")
        cls.fake_module.AsyncGroq = FakeGroq
        cls.dotenv_module = types.ModuleType("dotenv")
        cls.dotenv_module.load_dotenv = lambda *args, **kwargs: None
        cls.modules_patch = patch.dict(sys.modules, {"groq": cls.fake_module, "dotenv": cls.dotenv_module})
        cls.modules_patch.start()
        import groq_voice
        cls.voice = groq_voice

    @classmethod
    def tearDownClass(cls): cls.modules_patch.stop()

    def test_transcription_and_schema_parse(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "mock-key"}):
            transcript = asyncio.run(self.voice.transcribe(b"\x00\x00" * 30))
            parsed = asyncio.run(self.voice.parse_order("two cokes", [], [{"name": "Coke", "price": 1.99}]))
        self.assertEqual(transcript, "two cokes")
        self.assertEqual(parsed["tool"], "add_to_cart")
        transcription_api, parsing_api = FakeGroq.instances[-2:]
        self.assertEqual(transcription_api.audio.transcriptions_kwargs["model"], self.voice.STT_MODEL)
        self.assertEqual(parsing_api.chat_kwargs["model"], self.voice.LLM_MODEL)
        self.assertTrue(parsing_api.chat_kwargs["response_format"]["json_schema"]["strict"])
        self.assertEqual(self.voice.STT_MODEL, "whisper-large-v3-turbo")
        self.assertEqual(self.voice.LLM_MODEL, "openai/gpt-oss-120b")

    def test_long_tts_chunks_and_wav_assembly(self):
        text = ("pizza order ") * 45
        chunks = self.voice.split_tts_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))
        with patch.dict(os.environ, {"GROQ_API_KEY": "mock-key"}):
            data = asyncio.run(self.voice.synthesize(text))
        tts_calls = FakeGroq.instances[-1].audio.speech_calls
        self.assertEqual(tts_calls[0]["model"], self.voice.TTS_MODEL)
        self.assertEqual(tts_calls[0]["voice"], self.voice.TTS_VOICE)
        self.assertEqual(tts_calls[0]["response_format"], "wav")
        self.assertEqual(len(tts_calls), len(chunks))
        with wave.open(io.BytesIO(data), "rb") as wav:
            self.assertEqual(wav.getnframes(), sum(20 + index for index in range(1, len(chunks) + 1)))
            self.assertEqual(wav.getframerate(), 24000)

    def test_rate_limit_error_includes_retry_after_without_secrets(self):
        response = types.SimpleNamespace(headers={"retry-after": "12"})
        error = types.SimpleNamespace(status_code=429, response=response)
        message = self.voice.friendly_error(error)
        self.assertIn("Retry after 12 seconds", message)
        self.assertIn("rate limits and usage", message)


if __name__ == "__main__": unittest.main()
