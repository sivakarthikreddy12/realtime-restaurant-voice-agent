#!/usr/bin/env python3
"""Small local web UI for the realtime restaurant voice assistant.

The UI is a single HTML page served by FastAPI. A WebSocket bridge connects the
browser to the ``speech-to-speech serve`` Realtime backend, executes the local
restaurant tools, and streams transcripts plus PCM16 TTS audio back to the page.

Run:
    source .venv/bin/activate
    python ui_server.py

Then open http://127.0.0.1:8000.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from openai import AsyncOpenAI

import order_tools
from main import (
    DEFAULT_INSTRUCTIONS,
    NLTK_DATA_DIR,
    PROJECT_ROOT,
    _on_apple_silicon,
    configure_ssl_trust,
    ensure_nltk_data,
    validate_tool_module,
)
from speech_to_speech.api.openai_realtime.audio_client import (
    RealtimeAudioClientConfig,
    build_session_update,
    load_realtime_tool_module,
)

UI_INDEX = PROJECT_ROOT / "ui" / "index.html"
DEFAULT_UI_PORT = 8000
DEFAULT_S2S_PORT = 8765
SILENCE_SECONDS = 1.2
PCM_CHUNK_BYTES = 2048


def port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def wait_for_port(host: str, port: int, process: subprocess.Popen[Any] | None, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Voice backend exited with code {process.returncode}.")
        try:
            _reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Voice backend did not start on {host}:{port} within {timeout:.0f}s.")


def build_serve_argv(options: argparse.Namespace) -> list[str]:
    argv = ["serve"]
    if options.mac_optimal and _on_apple_silicon():
        argv.append("--mac-optimal-settings")
    argv.extend(
        [
            "--port",
            str(options.s2s_port),
            "--qwen3_tts_language",
            "en",
            "--init_chat_prompt",
            options.prompt,
        ]
    )
    if options.small_models:
        argv.extend(
            [
                "--stt",
                "mlx-audio-whisper",
                "--mlx_audio_whisper_model_name",
                "mlx-community/whisper-base-mlx",
                "--model_name",
                "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "--qwen3_tts_model_name",
                "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit",
            ]
        )
    argv.extend(options.extra_args)
    return argv


def start_voice_backend(options: argparse.Namespace) -> subprocess.Popen[Any] | None:
    if options.no_start:
        return None
    if port_open("127.0.0.1", options.s2s_port):
        raise RuntimeError(
            f"Port {options.s2s_port} is already in use. Stop the other server or pass "
            f"--s2s-port with a free port."
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["NLTK_DATA"] = str(NLTK_DATA_DIR)
    command = [sys.executable, "-m", "speech_to_speech.cli", *build_serve_argv(options)]
    print("Starting voice backend:")
    print("  " + " ".join(command), flush=True)
    return subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env)


def stop_voice_backend(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    print("Stopping voice backend...", flush=True)
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class BrowserSession:
    """One browser connection bridged to one Realtime backend session."""

    def __init__(self, browser: WebSocket, options: argparse.Namespace) -> None:
        self.browser = browser
        self.options = options
        self.pending_calls: list[dict[str, Any]] = []

    async def send(self, payload: dict[str, Any]) -> None:
        await self.browser.send_json(payload)

    async def run(self) -> None:
        order_tools.reset_state()
        tools, executor, create_response = load_realtime_tool_module("order_tools")
        config = RealtimeAudioClientConfig(
            tools=tools,
            tool_executor=executor,
            tool_response_create=create_response,
            send_rate=16000,
            recv_rate=16000,
            instructions=self.options.prompt,
        )
        client = AsyncOpenAI(
            base_url=f"http://127.0.0.1:{self.options.s2s_port}/v1",
            websocket_base_url=f"ws://127.0.0.1:{self.options.s2s_port}/v1",
            api_key="local",
        )
        try:
            async with client.realtime.connect(model="local") as conn:
                await conn.send(build_session_update(config))
                while True:
                    event = await asyncio.wait_for(conn.recv(), timeout=30)
                    if event.type == "session.updated":
                        break
                await self.send({"type": "ready"})

                browser_task = asyncio.create_task(self.pump_browser(conn))
                realtime_task = asyncio.create_task(self.pump_realtime(conn))
                done, pending = await asyncio.wait(
                    {browser_task, realtime_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    if not task.cancelled() and task.exception() is not None:
                        raise task.exception()  # type: ignore[misc]
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await self.send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            await client.close()

    async def pump_browser(self, conn: Any) -> None:
        while True:
            message = await self.browser.receive_json()
            kind = message.get("type")
            if kind == "text":
                text = str(message.get("text", "")).strip()
                if not text:
                    continue
                await conn.send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": text}],
                        },
                    }
                )
                await conn.send({"type": "response.create"})
                await self.send({"type": "thinking"})
            elif kind == "audio":
                audio = message.get("audio")
                if isinstance(audio, str) and audio:
                    await conn.send({"type": "input_audio_buffer.append", "audio": audio})
            elif kind == "audio_end":
                silence = b"\x00\x00" * int(16000 * SILENCE_SECONDS)
                for offset in range(0, len(silence), PCM_CHUNK_BYTES):
                    await conn.send(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                silence[offset : offset + PCM_CHUNK_BYTES]
                            ).decode("ascii"),
                        }
                    )
                await self.send({"type": "thinking"})
            elif kind == "mock_order":
                cart_result = await order_tools.execute_tool("get_cart", {})
                await self.send({"type": "tool_result", "name": "get_cart", "result": cart_result})
                if not cart_result.get("items"):
                    await self.send({"type": "notice", "text": "The cart is empty, so no mock order was created."})
                    continue
                receipt = await order_tools.execute_tool(
                    "create_mock_order", {"customer_confirmation": "yes, place the order"}
                )
                await self.send(
                    {"type": "tool_result", "name": "create_mock_order", "result": receipt}
                )
                if receipt.get("status") == "MOCK_ORDER_CREATED":
                    await self.send(
                        {
                            "type": "notice",
                            "text": f"Mock receipt {receipt['order_id']} created. No real order was placed.",
                        }
                    )
                else:
                    await self.send(
                        {
                            "type": "notice",
                            "text": receipt.get("error", "The mock order could not be created."),
                        }
                    )
            elif kind == "ping":
                await self.send({"type": "pong"})

    async def pump_realtime(self, conn: Any) -> None:
        while True:
            event = await conn.recv()
            kind = event.type

            if kind == "conversation.item.input_audio_transcription.completed":
                text = getattr(event, "transcript", "") or ""
                if text:
                    await self.send({"type": "user_transcript", "text": text})
            elif kind in {"response.output_audio_transcript.delta", "response.output_text.delta"}:
                await self.send({"type": "assistant_delta", "text": event.delta})
            elif kind == "response.output_audio_transcript.done":
                text = getattr(event, "transcript", "") or ""
                if text:
                    await self.send({"type": "assistant_done", "text": text})
            elif kind == "response.output_audio.delta":
                await self.send({"type": "audio", "audio": event.delta})
            elif kind == "response.output_audio.done":
                await self.send({"type": "audio_done"})
            elif kind == "response.created":
                await self.send({"type": "thinking"})
            elif kind == "response.function_call_arguments.done":
                try:
                    arguments = json.loads(event.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                call = {
                    "name": event.name,
                    "arguments": arguments,
                    "call_id": event.call_id,
                }
                self.pending_calls.append(call)
                await self.send(
                    {
                        "type": "tool_start",
                        "name": event.name,
                        "arguments": arguments,
                    }
                )
            elif kind == "response.done":
                status = getattr(event.response, "status", "completed")
                if self.pending_calls:
                    for call in self.pending_calls:
                        try:
                            result = await order_tools.execute_tool(call["name"], call["arguments"])
                        except Exception as exc:
                            result = {"error": f"{type(exc).__name__}: {exc}"}
                        await conn.send(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call["call_id"],
                                    "output": json.dumps(result),
                                },
                            }
                        )
                        await self.send(
                            {
                                "type": "tool_result",
                                "name": call["name"],
                                "result": result,
                            }
                        )
                    self.pending_calls.clear()
                    await conn.send({"type": "response.create"})
                else:
                    await self.send({"type": "response_done", "status": status})
            elif kind == "input_audio_buffer.speech_started":
                await self.send({"type": "listening"})
            elif kind == "input_audio_buffer.speech_stopped":
                await self.send({"type": "thinking"})
            elif kind == "error":
                error = getattr(event, "error", None)
                message = getattr(error, "message", None) or str(error)
                await self.send({"type": "error", "message": message})


@asynccontextmanager
async def lifespan(app: FastAPI):
    options: argparse.Namespace = app.state.options
    configure_ssl_trust()
    ensure_nltk_data()
    validate_tool_module("order_tools")

    process = start_voice_backend(options)
    app.state.s2s_process = process
    try:
        if options.no_start:
            await wait_for_port("127.0.0.1", options.s2s_port, None, 1.0)
        else:
            await wait_for_port("127.0.0.1", options.s2s_port, process, options.startup_timeout)
        yield
    finally:
        stop_voice_backend(process)


app = FastAPI(title="Demo Burger House Voice UI", lifespan=lifespan)
app.state.session_lock = asyncio.Lock()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(UI_INDEX, media_type="text/html")


@app.get("/healthz")
async def healthz() -> JSONResponse:
    options = app.state.options
    return JSONResponse(
        {
            "status": "ok",
            "backend": f"ws://127.0.0.1:{options.s2s_port}/v1/realtime",
            "ui": f"http://{options.host}:{options.ui_port}",
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    if app.state.session_lock.locked():
        await websocket.send_json(
            {
                "type": "error",
                "message": "Another browser session is active. Close the other tab or reload this page.",
            }
        )
        await websocket.close()
        return
    async with app.state.session_lock:
        await BrowserSession(websocket, app.state.options).run()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local web UI for the realtime restaurant voice assistant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="UI bind address (local only by default).")
    parser.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT, help="Browser UI port.")
    parser.add_argument("--s2s-port", type=int, default=DEFAULT_S2S_PORT, help="Internal Realtime server port.")
    parser.add_argument("--prompt", default=DEFAULT_INSTRUCTIONS, help="Assistant system prompt.")
    parser.add_argument(
        "--small-models",
        action="store_true",
        help="Use the smaller Whisper/Qwen2.5/Qwen3-TTS model set for a fast local demo.",
    )
    parser.add_argument("--no-start", action="store_true", help="Connect to an already-running Realtime server.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--startup-timeout", type=float, default=900.0, help="Seconds to wait for model startup.")
    parser.add_argument("--no-mac-optimal", action="store_true", help="Do not add --mac-optimal-settings.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra `speech-to-speech serve` arguments; put them after `--`.",
    )
    options = parser.parse_args(argv)
    extra = list(options.extra_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    options.extra_args = extra
    options.mac_optimal = not options.no_mac_optimal
    return options


def main(argv: list[str] | None = None) -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    options = parse_args(argv)
    app.state.options = options

    config = uvicorn.Config(app, host=options.host, port=options.ui_port, log_level="info")
    server = uvicorn.Server(config)
    url = f"http://{options.host}:{options.ui_port}"

    async def serve() -> None:
        task = asyncio.create_task(server.serve())
        while not server.started:
            if task.done():
                await task
                return
            await asyncio.sleep(0.1)
        print(f"\nWeb UI ready: {url}\n", flush=True)
        if not options.no_open:
            webbrowser.open(url)
        await task

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
