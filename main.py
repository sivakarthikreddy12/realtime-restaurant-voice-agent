#!/usr/bin/env python3
"""Launch the local realtime restaurant voice assistant.

This launcher fixes three setup problems that the bare ``speech-to-speech``
console command cannot solve on its own:

1. The project directory is not automatically on ``sys.path`` for an installed
   console script, so ``--tool-module order_tools`` fails with
   ``ModuleNotFoundError``. We add the project root before importing tools.
2. NLTK downloads its tokenizer data at import time and fails with
   ``CERTIFICATE_VERIFY_FAILED`` when the interpreter cannot find a CA bundle.
   We point Python at certifi's CA bundle *without disabling verification*.
3. Python 3.14 removed ``audioop`` from the standard library. The pinned
   ``speech-to-speech`` dependency chain provides ``audioop-lts``; the check
   command verifies that import before starting a session.

Usage:
    python main.py                 # start the microphone/speaker assistant
    python main.py --check         # verify environment, tools, and audio devices
    python main.py --mic-test      # record 1 second and print signal level
    python main.py --speaker-test  # play a short test tone
    python main.py -- --verbose    # pass extra flags to speech-to-speech local
"""

from __future__ import annotations

import argparse
import os
import platform
import ssl
import sys
from dataclasses import dataclass, field
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
NLTK_DATA_DIR = PROJECT_ROOT / ".nltk_data"
DEFAULT_TOOL_MODULE = "order_tools"
REQUIRED_NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}

DEFAULT_INSTRUCTIONS = (
    "You are a friendly English-speaking restaurant order-taking assistant for Demo Burger House. "
    "Always reply in English. Speak naturally and keep replies concise. "
    "Use the provided tools for the menu and cart, and never invent items or prices. "
    "When the customer changes the cart, call get_cart and read back every item, quantity, and the "
    "subtotal before asking them to confirm. "
    "When the customer explicitly says to place or confirm the order, you MUST call "
    "create_mock_order with the customer's exact words in customer_confirmation. "
    "Never say the order is placed unless create_mock_order returned status MOCK_ORDER_CREATED. "
    "This is a demo with sample data only. Never claim that a real Toast order was placed, and never "
    "call a real restaurant ordering API."
)


@dataclass
class LauncherOptions:
    prompt: str = DEFAULT_INSTRUCTIONS
    tool_module: str = DEFAULT_TOOL_MODULE
    input_device: int | None = None
    output_device: int | None = None
    port: int = 8765
    language: str = "en"
    block_mic_during_playback: bool = False
    verbose: bool = False
    print_json: bool = False
    mac_optimal: bool = True
    check: bool = False
    mic_test: bool = False
    speaker_test: bool = False
    extra_args: list[str] = field(default_factory=list)


def _on_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _default_ssl_context_has_certs() -> bool:
    try:
        context = ssl.create_default_context()
    except Exception:
        return False
    return bool(context.get_ca_certs())


def configure_ssl_trust() -> str | None:
    """Ensure a valid CA bundle is available while keeping TLS verification on.

    Returns the certifi bundle path when it was installed into the environment.
    Raises ``RuntimeError`` if the platform CA store is unusable and certifi is
    unavailable. This function never sets ``CERT_NONE`` or disables hostname
    checking.
    """

    if _default_ssl_context_has_certs():
        return None

    try:
        import certifi
    except ImportError as exc:  # pragma: no cover - requirements always install it
        raise RuntimeError(
            "No usable system CA certificates were found and certifi is not installed. "
            "Run: python -m pip install certifi"
        ) from exc

    bundle = certifi.where()
    if not os.path.isfile(bundle):
        raise RuntimeError(f"certifi CA bundle does not exist: {bundle}")

    os.environ["SSL_CERT_FILE"] = bundle
    os.environ["REQUESTS_CA_BUNDLE"] = bundle
    os.environ["CURL_CA_BUNDLE"] = bundle

    context = ssl.create_default_context(cafile=bundle)
    if not context.get_ca_certs():
        raise RuntimeError("The certifi CA bundle did not load any certificates.")
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise RuntimeError("TLS verification unexpectedly disabled; refusing to continue.")
    return bundle


def ensure_nltk_data(data_dir: Path = NLTK_DATA_DIR, *, download: bool = True) -> None:
    """Locate or safely download the NLTK resources used by speech-to-speech."""

    import nltk

    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = str(data_dir)
    if data_path not in nltk.data.path:
        nltk.data.path.insert(0, data_path)

    missing: list[tuple[str, str]] = []
    for package, resource in REQUIRED_NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource, paths=[data_path])
        except (LookupError, OSError):
            missing.append((package, resource))

    if not missing:
        return

    if not download:
        raise RuntimeError(
            "Missing NLTK data: "
            + ", ".join(resource for _, resource in missing)
            + f". Run `python -c \"import nltk; nltk.download('punkt_tab'); "
            "nltk.download('averaged_perceptron_tagger_eng')\"`."
        )

    for package, _resource in missing:
        print(f"Downloading NLTK package: {package}", flush=True)
        try:
            succeeded = nltk.download(package, download_dir=data_path, quiet=False)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download NLTK package {package!r}: {exc}. "
                "TLS certificate verification is still enabled. Check your network/proxy or "
                "install/update certifi with `python -m pip install --upgrade certifi`."
            ) from exc
        if not succeeded:
            raise RuntimeError(
                f"NLTK reported failure downloading {package!r}. "
                "TLS certificate verification is still enabled. Check your network/proxy or "
                "install/update certifi with `python -m pip install --upgrade certifi`."
            )

    for _package, resource in REQUIRED_NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource, paths=[data_path])
        except (LookupError, OSError) as exc:
            raise RuntimeError(f"NLTK resource is still missing after download: {resource}") from exc


def validate_tool_module(module_name: str = DEFAULT_TOOL_MODULE) -> tuple[list[dict[str, Any]], Any]:
    """Import the tool module and validate its tool schemas with the real client."""

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import_module(module_name)
    try:
        from speech_to_speech.api.openai_realtime.audio_client import load_realtime_tool_module
    except ImportError as exc:
        raise RuntimeError(
            "speech-to-speech is not installed in this interpreter. "
            "Activate .venv and run `python -m pip install -r requirements.txt`."
        ) from exc

    tools, executor, create_response = load_realtime_tool_module(module_name)
    if not callable(executor):
        raise RuntimeError(f"Tool module {module_name!r} does not expose a callable execute_tool().")
    if not tools:
        raise RuntimeError(f"Tool module {module_name!r} exposes no tools.")
    return tools, executor


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _check_audio_devices() -> tuple[Any, Any]:
    import sounddevice as sd

    input_info = sd.query_devices(kind="input")
    output_info = sd.query_devices(kind="output")
    if input_info["max_input_channels"] < 1:
        raise RuntimeError("The default audio input device has no microphone channels.")
    if output_info["max_output_channels"] < 1:
        raise RuntimeError("The default audio output device has no output channels.")

    sd.check_input_settings(samplerate=16000, channels=1, dtype="int16")
    sd.check_output_settings(samplerate=16000, channels=1, dtype="int16")
    return input_info, output_info


def check_environment(options: LauncherOptions) -> None:
    """Print a non-destructive environment check and raise on hard failures."""

    print("Restaurant voice agent environment check")
    print(f"  Python:             {platform.python_version()} ({sys.executable})")
    print(f"  Platform:           {platform.system()} {platform.machine()}")
    print(f"  speech-to-speech:   {_package_version('speech-to-speech')}")
    print(f"  NLTK:               {_package_version('nltk')}")
    print(f"  certifi:            {_package_version('certifi')}")
    print(f"  audioop:            {_package_version('audioop-lts')} (Python 3.13+ compatibility)")

    context = ssl.create_default_context()
    cert_count = len(context.get_ca_certs())
    if not cert_count or context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise RuntimeError(
            f"TLS verification is not healthy (CA certs={cert_count}, verify_mode={context.verify_mode})."
        )
    print(f"  TLS verification:   enabled ({cert_count} CA certificates)")

    tools, _ = validate_tool_module(options.tool_module)
    tool_names = ", ".join(tool["name"] for tool in tools)
    print(f"  Tool module:        {options.tool_module} ({len(tools)} tools: {tool_names})")
    print(f"  NLTK data:          {NLTK_DATA_DIR}")

    input_info, output_info = _check_audio_devices()
    print(f"  Microphone:         {input_info['name']} ({input_info['max_input_channels']} channel(s))")
    print(f"  Speaker:            {output_info['name']} ({output_info['max_output_channels']} channel(s))")

    import sounddevice as sd

    print("  Audio devices (use these indices with --input-device/--output-device):")
    for index, device in enumerate(sd.query_devices()):
        print(
            f"    [{index}] in={device['max_input_channels']} out={device['max_output_channels']} "
            f"{device['name']}"
        )
    print("Environment check passed.")


def run_mic_test(seconds: float = 1.0, device: int | None = None) -> None:
    """Record a short sample so microphone permissions and signal can be verified."""

    import numpy as np
    import sounddevice as sd

    rate = 16000
    frames = max(1, int(rate * seconds))
    sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
    print(f"Recording {seconds:.1f}s from the microphone. Please speak now...", flush=True)
    recording = sd.rec(frames, samplerate=rate, channels=1, dtype="int16", device=device)
    sd.wait()
    if recording.shape[0] != frames:
        raise RuntimeError(f"Microphone returned {recording.shape[0]} frames; expected {frames}.")
    rms = float(np.sqrt(np.mean(recording.astype(np.float64) ** 2)))
    print(f"Microphone test passed: {frames} frames captured, RMS level {rms:.1f}.")
    if rms < 1.0:
        print("Warning: the recording is silent. Check microphone permission and input selection.")


def run_speaker_test(device: int | None = None) -> None:
    """Play a quiet 440 Hz tone to verify speaker output."""

    import numpy as np
    import sounddevice as sd

    rate = 16000
    seconds = 0.4
    amplitude = 0.08
    timeline = np.arange(int(rate * seconds), dtype=np.float64) / rate
    tone = (np.sin(2 * np.pi * 440 * timeline) * amplitude * 32767).astype(np.int16)
    sd.check_output_settings(device=device, samplerate=rate, channels=1, dtype="int16")
    print("Playing a short test tone...", flush=True)
    sd.play(tone, samplerate=rate, device=device)
    sd.wait()
    print("Speaker test passed.")


def build_s2s_argv(options: LauncherOptions) -> list[str]:
    """Build the ``speech-to-speech local`` argument list."""

    argv = ["local"]
    if options.mac_optimal and _on_apple_silicon():
        argv.append("--mac-optimal-settings")
    argv.extend(
        [
            "--tool-module",
            options.tool_module,
            "--init_chat_prompt",
            options.prompt,
            "--port",
            str(options.port),
            "--qwen3_tts_language",
            options.language,
        ]
    )
    if options.input_device is not None:
        argv.extend(["--local_audio_input_device", str(options.input_device)])
    if options.output_device is not None:
        argv.extend(["--local_audio_output_device", str(options.output_device)])
    if options.block_mic_during_playback:
        argv.append("--local_audio_block_mic_during_playback")
    if options.verbose:
        argv.append("--verbose")
    if options.print_json:
        argv.append("--local_audio_print_json")
    argv.extend(options.extra_args)
    return argv


def run_local(options: LauncherOptions) -> None:
    """Start the realtime microphone/speaker pipeline via speech-to-speech."""

    try:
        from speech_to_speech.cli import main as speech_to_speech_main
    except ImportError as exc:
        raise RuntimeError(
            "Could not import speech_to_speech. Activate .venv and run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    validate_tool_module(options.tool_module)
    argv = build_s2s_argv(options)
    if "-h" not in argv and "--help" not in argv:
        print("Starting realtime restaurant voice assistant (English).", flush=True)
        print("Press Ctrl+C to stop. Headphones are recommended to prevent echo.", flush=True)
    original_argv = sys.argv
    sys.argv = [original_argv[0], *argv]
    try:
        speech_to_speech_main()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        sys.argv = original_argv


def parse_args(argv: list[str] | None = None) -> LauncherOptions:
    parser = argparse.ArgumentParser(
        description="Launch the local realtime restaurant voice assistant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", default=DEFAULT_INSTRUCTIONS, help="System prompt for the assistant.")
    parser.add_argument("--tool-module", default=DEFAULT_TOOL_MODULE, help="Importable tool module name.")
    parser.add_argument("--input-device", type=int, default=None, help="sounddevice microphone index.")
    parser.add_argument("--output-device", type=int, default=None, help="sounddevice speaker index.")
    parser.add_argument("--port", type=int, default=8765, help="Loopback port for the Realtime server.")
    parser.add_argument("--language", default="en", help="TTS language code (English by default).")
    parser.add_argument(
        "--block-mic-during-playback",
        action="store_true",
        help="Mute the microphone while the assistant speaks (use if echo is unavoidable).",
    )
    parser.add_argument("--verbose", action="store_true", help="Show routine logs and tool details.")
    parser.add_argument("--print-json", action="store_true", help="Print raw Realtime events.")
    parser.add_argument(
        "--no-mac-optimal",
        action="store_true",
        help="Do not add --mac-optimal-settings on Apple Silicon.",
    )
    parser.add_argument("--check", action="store_true", help="Check the environment and exit.")
    parser.add_argument("--mic-test", action="store_true", help="Record 1 second and print the signal level.")
    parser.add_argument("--speaker-test", action="store_true", help="Play a short test tone.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments for `speech-to-speech local`; put them after `--`.",
    )
    namespace = parser.parse_args(argv)
    extra = list(namespace.extra_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return LauncherOptions(
        prompt=namespace.prompt,
        tool_module=namespace.tool_module,
        input_device=namespace.input_device,
        output_device=namespace.output_device,
        port=namespace.port,
        language=namespace.language,
        block_mic_during_playback=namespace.block_mic_during_playback,
        verbose=namespace.verbose,
        print_json=namespace.print_json,
        mac_optimal=not namespace.no_mac_optimal,
        check=namespace.check,
        mic_test=namespace.mic_test,
        speaker_test=namespace.speaker_test,
        extra_args=extra,
    )


def main(argv: list[str] | None = None) -> int:
    # This must happen before importing speech_to_speech, because that package
    # imports NLTK and may download tokenizer data during module import.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    options = parse_args(argv)
    try:
        bundle = configure_ssl_trust()
        if bundle:
            print(f"Using certifi CA bundle for NLTK/TLS: {bundle}", flush=True)
        ensure_nltk_data()
        validate_tool_module(options.tool_module)

        if options.check:
            check_environment(options)
        if options.mic_test:
            run_mic_test(device=options.input_device)
        if options.speaker_test:
            run_speaker_test(device=options.output_device)
        if options.check or options.mic_test or options.speaker_test:
            return 0

        run_local(options)
        return 0
    except RuntimeError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
