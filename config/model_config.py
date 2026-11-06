"""Model and device settings grouped by supported runtime profile."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    stt_backend: str | None = None
    stt_model: str | None = None
    stt_device: str | None = None
    stt_compute_type: str | None = None
    llm_backend: str | None = None
    llm_model: str | None = None
    llm_device: str | None = None
    llm_dtype: str | None = None
    llm_max_new_tokens: int | None = None
    tts_backend: str | None = None
    tts_device: str | None = None
    tts_language: str | None = None
    tts_voice: str | None = None
    torch_num_threads: int | None = None

    def cli_args(self) -> list[str]:
        """Build the speech-to-speech CLI flags for this profile."""
        values = (
            ("--stt", self.stt_backend),
            ("--faster_whisper_stt_model_name", self.stt_model if self.stt_backend == "faster-whisper" else None),
            ("--faster_whisper_stt_device", self.stt_device if self.stt_backend == "faster-whisper" else None),
            (
                "--faster_whisper_stt_compute_type",
                self.stt_compute_type if self.stt_backend == "faster-whisper" else None,
            ),
            ("--llm_backend", self.llm_backend),
            ("--model_name", self.llm_model),
            ("--llm_device", self.llm_device),
            ("--llm_torch_dtype", self.llm_dtype),
            (
                "--llm_gen_max_new_tokens",
                str(self.llm_max_new_tokens) if self.llm_max_new_tokens is not None else None,
            ),
            ("--tts", self.tts_backend),
            ("--kokoro_device", self.tts_device if self.tts_backend == "kokoro" else None),
            ("--kokoro_lang_code", self.tts_language if self.tts_backend == "kokoro" else None),
            ("--kokoro_voice", self.tts_voice if self.tts_backend == "kokoro" else None),
            ("--torch_num_threads", str(self.torch_num_threads) if self.torch_num_threads is not None else None),
        )
        args: list[str] = []
        for flag, value in values:
            if value is not None:
                args.extend((flag, value))
        return args


WINDOWS_CPU_PROFILE = ModelProfile(
    name="windows-cpu",
    stt_backend="faster-whisper",
    stt_model="tiny.en",
    stt_device="cpu",
    stt_compute_type="int8",
    llm_backend="transformers",
    llm_model="Qwen/Qwen2.5-1.5B-Instruct",
    llm_device="cpu",
    llm_dtype="float16",
    llm_max_new_tokens=256,
    tts_backend="kokoro",
    tts_device="cpu",
    tts_language="a",
    tts_voice="af_heart",
    torch_num_threads=max(1, min(4, os.cpu_count() or 1)),
)

MACOS_APPLE_SILICON_PROFILE = ModelProfile(name="macos-apple-silicon")
FRAMEWORK_DEFAULTS_PROFILE = ModelProfile(name="framework-defaults")


def get_model_profile(name: str) -> ModelProfile:
    profiles = {
        "windows-cpu": WINDOWS_CPU_PROFILE,
        "macos-apple-silicon": MACOS_APPLE_SILICON_PROFILE,
        "framework-defaults": FRAMEWORK_DEFAULTS_PROFILE,
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"No model settings are defined for profile {name!r}.") from exc
