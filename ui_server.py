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
from difflib import SequenceMatcher
import json
import os
import re
import socket
import subprocess
import sys
import time
import webbrowser
from contextlib import asynccontextmanager
from typing import Any,cast

import uvicorn
import certifi
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
LATENCY_LOG = PROJECT_ROOT / "latency.log"
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


def build_browser_session_update(
    config: RealtimeAudioClientConfig,
) -> dict[str, Any]:
    session_update = build_session_update(config)

    turn_detection = session_update["session"]["audio"]["input"]["turn_detection"]
    turn_detection.update(
        {
            "create_response": True,
            "silence_duration_ms": 700,
        }
    )

    return session_update


_ORDER_TURN_RE = re.compile(
    r"\b(menu|cart|order|add|added|remove|removed|cancel|replace|change|quantity|"
    r"confirm|place|submit|burger|fries|coke|cola|milkshake|milk|shake|vanilla|"
    r"coffee|pizza|i'd like|i would like|i want|can i get|i'll have|give me)\b",
    re.IGNORECASE,
)
_MENU_REQUEST_RE = re.compile(r"\b(menu|what(?:'s| is| do you have).*available|what do you have)\b", re.I)
_CART_READ_RE = re.compile(
    r"\b(cart|subtotal|total|confirm|confirmation|review|what do i have|what's in my)\b",
    re.I,
)
_CONFIRMATION_QUESTION_RE = re.compile(
    r"\b(can|could|would) you (please )?(confirm|place|submit)|\bwhat (is|are) (in|on) my order\b",
    re.I,
)
_CLEAR_YES_RE = re.compile(
    r"^(yes please|yes place the order|yes confirm|"
    r"go ahead and place the order|place it|place the order|submit it|confirm it|"
    r"i confirm|i confirm this one|okay i confirm|i confirm the order|"
    r"yes please place the order|please place the order)$",
)


def is_order_related_turn(text: str) -> bool:
    return bool(_ORDER_TURN_RE.search(text or ""))


def _is_clear_yes(text: str) -> bool:
    return bool(_CLEAR_YES_RE.fullmatch(order_tools._normalize(text)))


def _is_clear_plan_yes(text: str) -> bool:
    return order_tools._is_plan_approval(text) and "?" not in text


def _mentions_item(text: str, item_name: str) -> bool:
    normalized_text = order_tools._normalize(text)
    matching_item = next(
        (item for item in order_tools.MENU if item_name.casefold() == item["name"].casefold()),
        None,
    )
    candidates = [item_name]
    if matching_item is not None:
        # Do not let a generic alias such as "burger" authorize Cheeseburger
        # when the customer actually said "Chicken Burger".
        other_names = [
            order_tools._normalize(item["name"])
            for item in order_tools.MENU
            if item is not matching_item
        ]
        if any(f" {name} " in f" {normalized_text} " for name in other_names):
            candidates = [matching_item["name"]]
        else:
            candidates = [matching_item["name"], *matching_item["aliases"]]
    for candidate in candidates:
        normalized_candidate = order_tools._normalize(candidate)
        if not normalized_candidate:
            continue
        if f" {normalized_candidate} " in f" {normalized_text} ":
            return True
        candidate_tokens = normalized_candidate.split()
        text_tokens = normalized_text.split()
        if len(candidate_tokens) > 1 and any(token in text_tokens for token in candidate_tokens):
            return True
        if len(candidate_tokens) == 1 and any(
            SequenceMatcher(None, candidate_tokens[0], token).ratio() >= 0.78
            for token in text_tokens
        ):
            return True
    return False


def cart_plan_matches_user_turn(user_text: str, arguments: dict[str, Any]) -> bool:
    """Ground plan entities in the latest turn; action semantics come from the schema."""
    operations = arguments.get("operations")
    if not isinstance(operations, list):
        return False
    cart = order_tools.CART
    for op in operations:
        if not isinstance(op, dict):
            return False
        pairs = [(str(op.get("item_name", "")), str(op.get("item_evidence", "")))]
        if op.get("action") == "replace":
            pairs.append((str(op.get("replacement_item_name", "")), str(op.get("replacement_item_evidence", ""))))
        for item_name, evidence in pairs:
            if not item_name or not evidence:
                return False
            norm_evidence = order_tools._normalize(evidence)
            norm_turn = order_tools._normalize(user_text)
            pronoun = norm_evidence in {"it", "that", "that item", "this one", "that one"}
            if pronoun:
                if len(cart) != 1 or order_tools._normalize(next(iter(cart))) != order_tools._normalize(item_name):
                    return False
            elif norm_evidence not in norm_turn:
                return False
            # Evidence may be an ASR spelling; require the chosen item to agree with
            # a known menu label/alias, unless it is explicitly an unavailable item.
            resolved = order_tools._find_item(item_name)
            if resolved and not _mentions_item(evidence, resolved["name"]):
                return False
            if not resolved:
                requested_name = order_tools._normalize(item_name)
                if requested_name not in norm_evidence and norm_evidence not in requested_name:
                    if SequenceMatcher(None, requested_name, norm_evidence).ratio() < 0.76:
                        return False
        action = op.get("action")
        quantity = op.get("quantity", 1)
        quantity_evidence = order_tools._normalize(op.get("quantity_evidence", ""))
        if action in {"add", "increase", "decrease", "set_quantity"}:
            if quantity_evidence:
                if quantity_evidence not in norm_turn:
                    return False
                word_numbers = {
                    "one": 1, "a": 1, "an": 1, "two": 2, "three": 3,
                    "four": 4, "five": 5, "six": 6, "seven": 7,
                    "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                    "twelve": 12, "thirteen": 13, "fourteen": 14,
                    "fifteen": 15, "sixteen": 16, "seventeen": 17,
                    "eighteen": 18, "nineteen": 19, "twenty": 20,
                    "pair": 2, "couple": 2,
                }
                amount = next(
                    (word_numbers[token] for token in quantity_evidence.split() if token in word_numbers),
                    None,
                )
                if amount is None:
                    amount = next(
                        (int(token) for token in quantity_evidence.split() if token.isdigit()),
                        None,
                    )
                if amount is None or amount != quantity:
                    return False
            elif quantity != 1 and not (action == "set_quantity" and quantity == 0):
                return False
            elif action in {"decrease", "set_quantity"} and not (action == "set_quantity" and quantity == 0):
                return False
    return True


def tool_matches_user_intent(
    tool_name: str,
    user_text: str,
    arguments: dict[str, Any] | None = None,
) -> bool:
    """Gate read-only tools and final order placement against the current turn."""
    text = user_text or ""
    if tool_name == "show_menu":
        return True  # Reading menu data has no side effects.
    if tool_name == "get_cart":
        return bool(_CART_READ_RE.search(text) or _ORDER_TURN_RE.search(text))
    if tool_name == "create_mock_order":
        return bool(_is_clear_yes(text) and not _CONFIRMATION_QUESTION_RE.search(text))
    return False


def format_verified_tool_response(
    calls: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    cart_result: dict[str, Any] | None = None,
) -> str:
    """Create customer-facing text from tool data instead of model paraphrases."""
    for call, result in zip(calls, results):
        if call.get("name") == "create_mock_order":
            if result.get("status") == "MOCK_ORDER_CREATED":
                return (
                    f"Your demo order {result['order_id']} is confirmed. "
                    f"The subtotal is ${result['subtotal']:.2f}. No real restaurant order was placed."
                )
            return f"I couldn't place the order: {result.get('error', 'the order was not created.')}"

    plan_result = next((result for call, result in zip(calls, results)
                        if call.get("name") in {"apply_cart_plan", "confirm_cart_plan"}), None)
    if plan_result is not None:
        status = plan_result.get("status")
        if status == "clarification_required":
            return str(plan_result.get("question") or "Could you clarify what change you want?")
        if status == "confirmation_required":
            changes = []
            for op in plan_result.get("planned_operations", []):
                action, name, qty = op.get("action"), op.get("item_name"), op.get("quantity")
                if action == "replace":
                    changes.append(f"replace {name} with {op.get('replacement_item_name')}")
                elif action in {"remove", "decrease"}:
                    changes.append(f"remove {qty} {name}" if action == "decrease" else f"remove {name}")
                elif action in {"add", "increase"}:
                    changes.append(f"add {qty} {name}")
                else:
                    changes.append(f"set {name} quantity to {qty}")
            proposal = "I understood: " + "; ".join(changes) + "."
            unavailable = [
                f"{entry.get('item_name', 'That item')} is unavailable"
                for entry in plan_result.get("invalid_items", [])
                if isinstance(entry, dict)
            ]
            if unavailable:
                proposal += " " + "; ".join(unavailable) + "."
            return proposal + " I have not changed the cart yet. Should I make these changes?"

    menu_result = next(
        (result for call, result in zip(calls, results) if call.get("name") == "show_menu"),
        None,
    )
    if menu_result is not None and isinstance(menu_result.get("menu"), list):
        entries = ", ".join(
            f"{item['name']} for ${float(item['price']):.2f}"
            for item in menu_result["menu"]
            if isinstance(item, dict) and "name" in item and "price" in item
        )
        return f"Our menu is: {entries}. What would you like to order?"

    problems: list[str] = []
    for result in results:
        error = result.get("error")
        if error:
            problems.append(str(error))
        for invalid in result.get("invalid_items", []):
            if isinstance(invalid, dict):
                problems.append(
                    f"{invalid.get('item_name', 'That item')} was not added: {invalid.get('reason', 'it is unavailable.')}"
                )
    for result in results:
        if result.get("status") == "no_change" and result.get("error"):
            problems.append(str(result["error"]))

    if cart_result is not None:
        items = cart_result.get("items", cart_result.get("cart", []))
        subtotal = float(cart_result.get("subtotal", 0.0))
        if items:
            lines = "; ".join(
                f"{int(item['quantity'])} {item['name']} (${float(item.get('line_total', item.get('unit_price', 0) * item['quantity'])):.2f})"
                for item in items
            )
            cart_text = f"Your current cart is: {lines}. Subtotal: ${subtotal:.2f}."
        else:
            cart_text = "Your cart is empty."
        if problems:
            response = " ".join(problems) + " " + cart_text
            available = next((r.get("available_items") for r in results if r.get("available_items")), None)
            if available:
                response += " Available menu items: " + ", ".join(available) + "."
            if items:
                response += (
                    " Would you like to add anything else? If everything is correct, "
                    "say ‘Yes, place the order.’"
                )
            return response
        return cart_text + " Would you like to add anything else or confirm the order?"

    if problems:
        available = next((r.get("available_items") for r in results if r.get("available_items")), None)
        response = " ".join(problems)
        if available:
            response += " Available menu items: " + ", ".join(available) + "."
        return response
    return "I couldn't verify that request, so I made no change to your cart."


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
    ca_bundle = certifi.where()
    env["SSL_CERT_FILE"] = ca_bundle
    env["REQUESTS_CA_BUNDLE"] = ca_bundle
    env["CURL_CA_BUNDLE"] = ca_bundle
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
        self.latest_user_text = ""
        self.active_response_user_text = ""
        self.buffer_order_response = False
        self.buffered_assistant_text: list[str] = []
        self.buffered_audio: list[str] = []
        self.awaiting_order_confirmation = False
        self.latency_turn = 0
        self.speech_started_at: float | None = None
        self.speech_ended_at: float | None = None
        self.response_started_at: float | None = None
        self.response_cycle = 0
        self.first_text_logged = False
        self.first_audio_logged = False

    def log_latency(self, stage: str, **measurements: float | int | str) -> None:
        details = " ".join(
            f"{name}={value:.0f}ms"
            if isinstance(value, (int, float))
            and ("duration" in name or name.startswith("after_") or name.endswith("_pause"))
            else f"{name}={value}"
            for name, value in measurements.items()
        )
        line = (
            f"[{time.strftime('%H:%M:%S')}] [LATENCY] turn={self.latency_turn} "
            f"{stage}{' ' + details if details else ''}"
        )
        print(line, flush=True)
        with LATENCY_LOG.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")

    @staticmethod
    def elapsed_ms(started_at: float, ended_at: float | None = None) -> float:
        return ((ended_at if ended_at is not None else time.perf_counter()) - started_at) * 1000

    async def send(self, payload: dict[str, Any]) -> None:
        await self.browser.send_json(payload)

    def _note_user_text(self, text: str) -> None:
        self.latest_user_text = text.strip()
        if not (_is_clear_yes(self.latest_user_text) or _is_clear_plan_yes(self.latest_user_text)):
            self.awaiting_order_confirmation = False
            order_tools.discard_pending_cart_plan()

    def _clear_response_buffer(self) -> None:
        self.buffered_assistant_text.clear()
        self.buffered_audio.clear()
        self.buffer_order_response = False

    async def _send_verified_response(self, text: str) -> None:
        self.log_latency("verified_response", characters=len(text))
        await self.send({"type": "verified_response", "text": text})

    async def _finish_response_without_tools(self) -> None:
        user_text = self.active_response_user_text
        if not self.buffer_order_response:
            await self.send({"type": "response_done", "status": "completed"})
            return

        if _MENU_REQUEST_RE.search(user_text):
            result = await order_tools.execute_tool("show_menu", {})
            await self.send({"type": "tool_result", "name": "show_menu", "result": result})
            await self._send_verified_response(
                format_verified_tool_response([{"name": "show_menu"}], [result])
            )
            return

        if order_tools.PENDING_CART_PLAN and _is_clear_plan_yes(user_text):
            result = await order_tools.execute_tool("confirm_cart_plan", {"confirmation": user_text})
            await self.send({"type": "tool_result", "name": "confirm_cart_plan", "result": result})
            cart_result = await order_tools.execute_tool("get_cart", {})
            await self.send({"type": "tool_result", "name": "get_cart", "result": cart_result})
            self.awaiting_order_confirmation = bool(cart_result.get("items"))
            await self._send_verified_response(format_verified_tool_response(
                [{"name": "confirm_cart_plan"}], [result], cart_result=cart_result
            ))
            return

        if _is_clear_yes(user_text) and not _CONFIRMATION_QUESTION_RE.search(user_text):
            if self.awaiting_order_confirmation and order_tools.PENDING_CONFIRMATION:
                result = await order_tools.execute_tool(
                    "create_mock_order", {"customer_confirmation": user_text}
                )
                self.awaiting_order_confirmation = False
                await self.send({"type": "tool_result", "name": "create_mock_order", "result": result})
                await self._send_verified_response(
                    format_verified_tool_response(
                        [{"name": "create_mock_order"}], [result]
                    )
                )
            else:
                result = await order_tools.execute_tool("get_cart", {})
                await self.send({"type": "tool_result", "name": "get_cart", "result": result})
                if result.get("items"):
                    text = format_verified_tool_response(
                        [{"name": "get_cart"}], [result], cart_result=result
                    )
                    text = text.replace(
                        "Would you like to add anything else or confirm the order?",
                        "Please review the order. If it is correct, say ‘Yes, place the order.’",
                    )
                    self.awaiting_order_confirmation = True
                else:
                    text = "Your cart is empty. Tell me what you would like to order."
                    self.awaiting_order_confirmation = False
                await self._send_verified_response(text)
            return

        if _CART_READ_RE.search(user_text):
            result = await order_tools.execute_tool("get_cart", {})
            await self.send({"type": "tool_result", "name": "get_cart", "result": result})
            text = format_verified_tool_response(
                [{"name": "get_cart"}], [result], cart_result=result
            )
            if result.get("items"):
                text = text.replace(
                    "Would you like to add anything else or confirm the order?",
                    "Please review the order. If it is correct, say ‘Yes, place the order.’",
                )
                self.awaiting_order_confirmation = True
            else:
                self.awaiting_order_confirmation = False
            await self._send_verified_response(text)
            return

        menu_names = ", ".join(item["name"] for item in order_tools.MENU)
        await self._send_verified_response(
            "I couldn't verify a cart action, so I haven't changed your order. "
            f"Available menu items are {menu_names}. Please tell me what you would like."
        )

    async def _finish_tool_response(self, conn: Any, calls: list[dict[str, Any]]) -> None:
        results: list[dict[str, Any]] = []
        for call in calls:
            tool_started_at = time.perf_counter()
            if call["name"] == "apply_cart_plan" and not call["arguments"].get("needs_clarification") and not cart_plan_matches_user_turn(self.active_response_user_text, call["arguments"]):
                result = {"status": "clarification_required", "question": "I couldn't match that item to what you said, so I made no cart change. Could you repeat the item?"}
            elif call["name"] == "confirm_cart_plan" and (not _is_clear_plan_yes(self.active_response_user_text) or not order_tools.PENDING_CART_PLAN):
                result = {"status": "no_pending_plan", "error": "There is no approved pending change to apply."}
            elif call["name"] not in {"apply_cart_plan", "confirm_cart_plan", "show_menu", "get_cart", "create_mock_order"}:
                result = {"error": "Individual cart mutation tools are disabled; use a structured cart plan."}
            elif call["name"] == "create_mock_order" and (not self.awaiting_order_confirmation or not _is_clear_yes(self.active_response_user_text)):
                result = {
                    "error": "Order confirmation has not been requested by the app; no order was placed."
                }
            elif not tool_matches_user_intent(
                call["name"], self.active_response_user_text, call["arguments"]
            ):
                result = {
                    "error": "This tool call did not match the customer's latest request; no change was made."
                }
            else:
                try:
                    arguments = call["arguments"]
                    if call["name"] == "confirm_cart_plan":
                        arguments = {"confirmation": self.active_response_user_text}
                    result = await order_tools.execute_tool(call["name"], arguments)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            self.log_latency(
                "tool_complete",
                name=call["name"],
                duration=self.elapsed_ms(tool_started_at),
            )
            results.append(result)
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
            await self.send({"type": "tool_result", "name": call["name"], "result": result})

        cart_result = None
        if any(
            call["name"] in {
                "add_to_cart", "remove_from_cart", "remove_item_from_cart",
                "set_item_quantity", "replace_cart_item", "get_cart", "apply_cart_plan", "confirm_cart_plan",
            }
            for call in calls
        ):
            cart_result = await order_tools.execute_tool("get_cart", {})
            await self.send({"type": "tool_result", "name": "get_cart", "result": cart_result})
        if any(r.get("status") == "confirmation_required" for r in results):
            self.awaiting_order_confirmation = False
        elif cart_result and cart_result.get("items"):
            self.awaiting_order_confirmation = True
        else:
            self.awaiting_order_confirmation = False
        for call, result in zip(calls, results):
            if call["name"] == "create_mock_order" and result.get("status") == "MOCK_ORDER_CREATED":
                self.awaiting_order_confirmation = False
        await self._send_verified_response(
            format_verified_tool_response(calls, results, cart_result=cart_result).replace(
                "Would you like to add anything else or confirm the order?",
                "Would you like to add anything else? If everything is correct, say ‘Yes, place the order.’",
            )
        )

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
                session_update = build_browser_session_update(config)
                await conn.send(cast(Any, session_update))
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
                self._note_user_text(text)
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
                if text.strip():
                    self._note_user_text(text)
                if self.speech_ended_at is not None:
                    self.log_latency(
                        "transcription_complete",
                        after_speech_end=self.elapsed_ms(self.speech_ended_at),
                    )
                if text:
                    await self.send({"type": "user_transcript", "text": text})
            elif kind in {"response.output_audio_transcript.delta", "response.output_text.delta"}:
                if self.buffer_order_response:
                    self.buffered_assistant_text.append(event.delta)
                    continue
                if self.response_started_at is not None and not self.first_text_logged:
                    self.first_text_logged = True
                    self.log_latency(
                        "first_text",
                        after_response_start=self.elapsed_ms(self.response_started_at),
                    )
                await self.send({"type": "assistant_delta", "text": event.delta})
            elif kind == "response.output_audio_transcript.done":
                text = getattr(event, "transcript", "") or ""
                if self.buffer_order_response:
                    if text and not self.buffered_assistant_text:
                        self.buffered_assistant_text.append(text)
                    continue
                if text:
                    await self.send({"type": "assistant_done", "text": text})
            elif kind == "response.output_audio.delta":
                if self.buffer_order_response:
                    self.buffered_audio.append(event.delta)
                    continue
                if self.response_started_at is not None and not self.first_audio_logged:
                    self.first_audio_logged = True
                    self.log_latency(
                        "first_audio",
                        after_response_start=self.elapsed_ms(self.response_started_at),
                        after_speech_end=self.elapsed_ms(self.speech_ended_at)
                        if self.speech_ended_at is not None
                        else "n/a",
                    )
                await self.send({"type": "audio", "audio": event.delta})
            elif kind == "response.output_audio.done":
                await self.send({"type": "audio_done"})
            elif kind == "response.created":
                self.response_cycle += 1
                self.response_started_at = time.perf_counter()
                self.active_response_user_text = self.latest_user_text
                self.buffer_order_response = is_order_related_turn(self.active_response_user_text)
                self.buffered_assistant_text.clear()
                self.buffered_audio.clear()
                self.first_text_logged = False
                self.first_audio_logged = False
                self.log_latency(
                    "response_started",
                    cycle=self.response_cycle,
                    after_speech_end=self.elapsed_ms(self.speech_ended_at)
                    if self.speech_ended_at is not None
                    else "n/a",
                )
                await self.send({"type": "response_started"})
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
                if self.response_started_at is not None:
                    self.log_latency(
                        "response_complete",
                        cycle=self.response_cycle,
                        duration=self.elapsed_ms(self.response_started_at),
                        status=status,
                    )
                calls = self.pending_calls
                self.pending_calls = []
                if status != "completed":
                    # A cancelled response must never commit an unfinished order action.
                    if self.buffer_order_response:
                        await self.send({"type": "response_done", "status": status})
                    else:
                        for audio in self.buffered_audio:
                            await self.send({"type": "audio", "audio": audio})
                        await self.send({"type": "response_done", "status": status})
                    self._clear_response_buffer()
                    continue

                if calls:
                    await self._finish_tool_response(conn, calls)
                else:
                    await self._finish_response_without_tools()
                self._clear_response_buffer()
            elif kind == "input_audio_buffer.speech_started":
                self.latency_turn += 1
                self.latest_user_text = ""
                self.speech_started_at = time.perf_counter()
                self.speech_ended_at = None
                self.response_cycle = 0
                self.log_latency("speech_started")
                await self.send({"type": "listening"})
            elif kind == "input_audio_buffer.speech_stopped":
                self.speech_ended_at = time.perf_counter()
                self.log_latency(
                    "speech_end_detected",
                    speech_duration=self.elapsed_ms(self.speech_started_at, self.speech_ended_at)
                    if self.speech_started_at is not None
                    else "n/a",
                    vad_pause=700,
                )
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
