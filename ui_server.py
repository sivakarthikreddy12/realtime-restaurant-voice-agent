#!/usr/bin/env python3
"""Local restaurant voice UI backed by Groq Cloud speech APIs."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import uuid
import webbrowser
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import groq_voice

import order_tools
from main import DEFAULT_INSTRUCTIONS, PROJECT_ROOT

UI_INDEX = PROJECT_ROOT / "ui" / "index.html"
DEFAULT_UI_PORT = 8000
UI_INDEX = PROJECT_ROOT / "ui" / "index.html"
DEFAULT_UI_PORT = 8000


class BrowserSession:
    def __init__(self, browser: WebSocket, options: argparse.Namespace) -> None:
        self.browser, self.options = browser, options
        self.session_id = str(uuid.uuid4()); self.store_id = "sanpeggios-demo"
        self.pcm = bytearray(); self.history: list[dict[str, str]] = []

    async def send(self, payload: dict[str, Any]) -> None: await self.browser.send_json(payload)

    async def run(self) -> None:
        requested = self.browser.query_params.get("session_id")
        if requested:
            try: self.session_id = str(uuid.UUID(requested))
            except ValueError: pass
        self.store_id = self.browser.query_params.get("store_id", "sanpeggios-demo")[:80]
        await order_tools.ensure_persisted_session(self.session_id, self.store_id)
        with order_tools.use_session(self.session_id, self.store_id):
            cart, subtotal = await order_tools.get_session_cart_snapshot(self.session_id)
            await self.send({"type":"cart_state", "items":cart, "subtotal":subtotal}); await self.send({"type":"ready"})
            welcome = await order_tools.session_welcome(self.session_id, self.store_id)
            if welcome: await self.reply(welcome)
            try:
                while True:
                    msg = await self.browser.receive_json(); kind = msg.get("type")
                    if kind == "audio":
                        audio = msg.get("audio")
                        if isinstance(audio, str): self.pcm.extend(base64.b64decode(audio))
                    elif kind == "audio_end":
                        audio = bytes(self.pcm); self.pcm.clear()
                        if audio: await self.handle_audio(audio)
                    elif kind == "text":
                        text = str(msg.get("text", "")).strip()
                        if text: await self.handle_text(text)
                    elif kind == "mock_order": await self.mock_order()
            except WebSocketDisconnect: pass
            except Exception as exc:
                try: await self.send({"type":"error", "message":str(exc)})
                except Exception: pass

    async def handle_audio(self, pcm: bytes) -> None:
        await self.send({"type":"thinking"})
        try:
            text = await groq_voice.transcribe(pcm)
            if not text:
                await self.send({"type":"notice", "text":"I didn't catch that. Please try again."}); await self.send({"type":"response_done"}); return
            await self.send({"type":"user_transcript", "text":text}); await self.handle_text(text)
        except Exception as exc: await self.send({"type":"error", "message":str(exc)})

    async def handle_text(self, text: str) -> None:
        await order_tools.invalidate_review_for_message(text); await self.send({"type":"thinking"})
        try:
            menu_result = await order_tools.execute_tool("show_menu", {})
            if menu_result.get("error"):
                await self.reply(str(menu_result["error"])); return
            menu = menu_result.get("menu", menu_result)
            parsed = await groq_voice.parse_order(text, self.history, menu)
            reply, tool = str(parsed.get("reply", "")), parsed.get("tool")
            if tool:
                args = parsed.get("arguments", {})
                if not isinstance(args, dict): args = {}
                await self.send({"type":"tool_start", "name":tool, "arguments":args})
                with order_tools.use_user_message(text): result = await order_tools.execute_tool(tool, args)
                await self.send({"type":"tool_result", "name":tool, "result":result})
                if result.get("error"): reply = str(result["error"])
                elif tool == "show_menu": pass
                else:
                    cart, subtotal = await order_tools.get_session_cart_snapshot(self.session_id)
                    await self.send({"type":"cart_state", "items":cart, "subtotal":subtotal})
                    if tool == "get_cart": reply = ("Your cart: " + ", ".join(f"{x['quantity']} {x['name']}" for x in cart) + f". Subtotal ${subtotal:.2f}.") if cart else "Your cart is empty."
                    elif not result.get("error"):
                        summary = ", ".join(f"{item['quantity']} {item['name']}" for item in cart)
                        reply = f"Your cart now has {summary}. Subtotal ${subtotal:.2f}."
            if not reply: reply = "What would you like to order?"
            self.history.extend([{"role":"user","content":text},{"role":"assistant","content":reply}]); await self.reply(reply)
        except Exception as exc: await self.send({"type":"error", "message":str(exc)})

    async def reply(self, text: str) -> None:
        await self.send({"type":"response_started"}); await self.send({"type":"assistant_delta", "text":text}); await self.send({"type":"assistant_done", "text":text})
        try: await self.send({"type":"audio", "audio":base64.b64encode(await groq_voice.synthesize(text)).decode("ascii"), "format":"wav"})
        except Exception as exc: await self.send({"type":"notice", "text":str(exc)})
        await self.send({"type":"audio_done"}); await self.send({"type":"response_done"})

    async def mock_order(self) -> None:
        result = await order_tools.execute_tool("get_cart", {}); await self.send({"type":"tool_result", "name":"get_cart", "result":result})
        if not result.get("items"):
            await self.send({"type":"notice", "text":"The cart is empty, so no mock order was created."}); return
        receipt = await order_tools.execute_tool("create_mock_order", {"customer_confirmation":"yes, place the order"}); await self.send({"type":"tool_result", "name":"create_mock_order", "result":receipt})
        if receipt.get("status") == "MOCK_ORDER_CREATED": await self.send({"type":"notice", "text":f"Mock receipt {receipt['order_id']} created. No real order was placed."})
        else: await self.send({"type":"notice", "text":receipt.get("error", "The mock order could not be created.")})


@asynccontextmanager
async def lifespan(app: FastAPI):
    options: argparse.Namespace = app.state.options
    import postgres_store

    app.state.database_warning = None
    try:
        await postgres_store.open_store(order_tools.MENU)
    except Exception as exc:
        postgres_store.mark_unavailable()
        app.state.database_warning = (
            "PostgreSQL is unavailable; using the built-in sample menu and in-memory carts. "
            "Start PostgreSQL and restart the app to enable persistence."
        )
        print(f"{app.state.database_warning} ({type(exc).__name__})", flush=True)
    if postgres_store.is_open():
        import menu_sync
        result = await menu_sync.sync_store()
        if result.get("status") == "success":
            order_tools._STORE_MENU_CACHE.pop("sanpeggios-demo", None)

    try:
        yield
    finally:
        await postgres_store.close_store()


app = FastAPI(title="Sanpeggio's Pizza Voice Ordering Demo", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(UI_INDEX, media_type="text/html")


@app.get("/healthz")
async def healthz() -> JSONResponse:
    options = app.state.options
    import postgres_store
    return JSONResponse(
        {
            "status": "ok",
            "backend": "Groq Cloud API",
            "ui": f"http://{options.host}:{options.ui_port}",
            "database": "connected" if postgres_store.is_open() else "memory_fallback",
            "database_warning": getattr(app.state, "database_warning", None),
        }
    )


@app.post("/menu/sync")
async def sync_menu(store_id: str = "sanpeggios-demo") -> JSONResponse:
    import menu_sync
    result = await menu_sync.sync_store(store_id)
    order_tools._STORE_MENU_CACHE.pop(store_id, None)
    return JSONResponse(result, status_code=200 if result.get("status") == "success" else 502)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    await BrowserSession(websocket, app.state.options).run()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local web UI for the realtime restaurant voice assistant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="UI bind address (local only by default).")
    parser.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT, help="Browser UI port.")
    parser.add_argument("--prompt", default=DEFAULT_INSTRUCTIONS, help="Assistant system prompt.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    options = parser.parse_args(argv)
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
