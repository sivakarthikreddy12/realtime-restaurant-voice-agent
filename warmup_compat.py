"""Project-level safeguards for speech-to-speech Transformers warmup."""

from __future__ import annotations

import importlib
import logging
import time
from queue import Empty
from threading import Thread
from typing import Any, Callable

from memory_monitor import log_memory_snapshot

logger = logging.getLogger(__name__)

WARMUP_MAX_SECONDS = 120.0
WARMUP_PROBE_TOKENS = 4
_PATCH_MARKER = "_restaurant_agent_warmup_patch"


def drain_streamer_while_worker_runs(
    streamer: Any,
    worker: Thread,
    worker_errors: list[BaseException],
    *,
    timeout_seconds: float = WARMUP_MAX_SECONDS,
    cancel: Callable[[], None] | None = None,
) -> None:
    """Wait through empty queue polls and preserve the producer's actual error."""
    deadline = time.monotonic() + timeout_seconds
    iterator = iter(streamer)
    completed = False
    while time.monotonic() < deadline:
        try:
            next(iterator)
        except StopIteration:
            completed = True
            break
        except Empty:
            if not worker.is_alive():
                break

    if not completed and worker.is_alive():
        if cancel is not None:
            cancel()
        worker.join(timeout=5.0)
        raise TimeoutError(
            f"LLM warmup did not finish within {timeout_seconds:.0f}s; generation was cancelled."
        )

    worker.join(timeout=5.0)
    if worker_errors:
        error = worker_errors[0]
        raise RuntimeError(
            f"LLM warmup generation worker failed: {type(error).__name__}: {error}"
        ) from error
    if worker.is_alive():
        raise TimeoutError("LLM warmup generation worker did not stop after closing its stream.")
    if not completed:
        raise RuntimeError("LLM warmup worker exited without closing the text stream.")


def _instrument_setup(module_name: str, class_name: str, stage: str) -> None:
    module = importlib.import_module(module_name)
    handler_class = getattr(module, class_name)
    setup = handler_class.setup
    marker = f"_restaurant_agent_memory_{stage}"
    if getattr(handler_class, marker, False):
        return

    def measured_setup(handler: Any, *args: Any, **kwargs: Any) -> Any:
        model_name = kwargs.get("model_name", "configured")
        log_memory_snapshot(f"{stage}_before:{model_name}")
        result = setup(handler, *args, **kwargs)
        log_memory_snapshot(f"{stage}_after:{model_name}")
        return result

    handler_class.setup = measured_setup
    setattr(handler_class, marker, True)


def _install_language_model_patch() -> None:
    from speech_to_speech.LLM.language_model import LanguageModelHandler

    if getattr(LanguageModelHandler, _PATCH_MARKER, False):
        return

    original_warmup = LanguageModelHandler.warmup
    original_load_model = LanguageModelHandler._load_model

    def measured_load_model(handler: Any, model_name: str, *args: Any, **kwargs: Any) -> Any:
        log_memory_snapshot(f"llm_before_load:{model_name}")
        result = original_load_model(handler, model_name, *args, **kwargs)
        log_memory_snapshot(f"llm_after_load:{model_name}")
        return result

    def safe_warmup(handler: Any) -> None:
        if handler.backend != "transformers":
            return original_warmup(handler)

        model_name = getattr(handler, "model_name", "unknown")
        logger.info("Warming up Transformers LLM with a %s-token probe", WARMUP_PROBE_TOKENS)
        dummy_chat = [{"role": handler.user_role, "content": "Repeat the word 'home'."}]
        generation_kwargs = {
            **handler.gen_kwargs,
            "min_new_tokens": 0,
            "max_new_tokens": min(
                WARMUP_PROBE_TOKENS,
                max(1, int(handler.gen_kwargs.get("max_new_tokens", WARMUP_PROBE_TOKENS))),
            ),
        }
        handler._cancel_criteria.reset()
        errors: list[BaseException] = []

        def generate() -> None:
            try:
                handler.pipe(dummy_chat, **generation_kwargs)
            except BaseException as exc:
                errors.append(exc)

        worker = Thread(target=generate, name="restaurant-llm-warmup", daemon=True)
        worker.start()
        try:
            drain_streamer_while_worker_runs(
                handler.streamer,
                worker,
                errors,
                cancel=handler._cancel_criteria.cancel,
            )
        except Exception:
            handler._cancel_criteria.cancel()
            raise
        log_memory_snapshot(f"llm_warmup_complete:{model_name}")
        logger.info("Transformers LLM warmup completed")

    LanguageModelHandler._load_model = measured_load_model
    LanguageModelHandler.warmup = safe_warmup
    setattr(LanguageModelHandler, "_restaurant_original_warmup", original_warmup)
    setattr(LanguageModelHandler, _PATCH_MARKER, True)


def install_framework_compatibility(*, windows_cpu: bool = False) -> None:
    """Patch runtime methods in memory; no installed package files are changed."""
    _install_language_model_patch()
    if windows_cpu:
        try:
            _instrument_setup(
                "speech_to_speech.STT.faster_whisper_handler",
                "FasterWhisperSTTHandler",
                "stt",
            )
            _instrument_setup(
                "speech_to_speech.TTS.kokoro_handler",
                "KokoroTTSHandler",
                "tts",
            )
        except ImportError as exc:
            raise RuntimeError(
                "The Windows CPU profile requires the optional Faster-Whisper and Kokoro "
                "dependencies. Install them with `python -m pip install -r requirements-windows.txt`."
            ) from exc
