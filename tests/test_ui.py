"""Tests for the local web UI launcher and static page."""

from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui_server  # noqa: E402


class WebUILauncherTests(unittest.TestCase):
    def test_serve_argv_has_english_and_small_model_overrides(self) -> None:
        options = ui_server.parse_args(
            ["--small-models", "--s2s-port", "9001", "--", "--log_level", "warning"]
        )
        argv = ui_server.build_serve_argv(options)
        self.assertEqual(argv[0], "serve")
        self.assertIn("--port", argv)
        self.assertEqual(argv[argv.index("--port") + 1], "9001")
        self.assertIn("--qwen3_tts_language", argv)
        self.assertEqual(argv[argv.index("--qwen3_tts_language") + 1], "en")
        self.assertIn("--stt", argv)
        self.assertEqual(argv[argv.index("--stt") + 1], "mlx-audio-whisper")
        self.assertIn("mlx-community/Qwen2.5-1.5B-Instruct-4bit", argv)
        self.assertEqual(argv[-2:], ["--log_level", "warning"])

    def test_port_open_detects_free_port(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            free_port = sock.getsockname()[1]
        self.assertFalse(ui_server.port_open("127.0.0.1", free_port))

    def test_static_page_has_required_controls(self) -> None:
        self.assertTrue(ui_server.UI_INDEX.exists())
        html = ui_server.UI_INDEX.read_text(encoding="utf-8")
        for token in (
            'id="chat"',
            'id="micBtn"',
            'id="textInput"',
            'id="sendBtn"',
            'id="cart"',
            'id="subtotal"',
            'id="tools"',
            'id="mockOrderBtn"',
            "navigator.mediaDevices.getUserMedia",
            "AudioWorkletNode",
            "type: \"audio\"",
            "type: \"text\"",
        ):
            with self.subTest(token=token):
                self.assertIn(token, html)

    def test_routes_are_registered(self) -> None:
        paths = {getattr(route, "path", None) for route in ui_server.app.routes}
        self.assertIn("/", paths)
        self.assertIn("/ws", paths)
        self.assertIn("/healthz", paths)

    def test_prompt_requires_mock_order_tool(self) -> None:
        self.assertIn("MUST call", ui_server.DEFAULT_INSTRUCTIONS)
        self.assertIn("create_mock_order", ui_server.DEFAULT_INSTRUCTIONS)
        self.assertIn("MOCK_ORDER_CREATED", ui_server.DEFAULT_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
