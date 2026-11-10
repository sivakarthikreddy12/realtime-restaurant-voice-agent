"""Tests for the local web UI launcher and static page."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui_server  # noqa: E402


class WebUILauncherTests(unittest.TestCase):
    def test_options_are_for_local_ui_only(self) -> None:
        options = ui_server.parse_args(["--ui-port", "8001", "--no-open"])
        self.assertEqual(options.ui_port, 8001)
        self.assertTrue(options.no_open)
        with self.assertRaises(SystemExit):
            ui_server.parse_args(["--small-models"])

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
            "function stopPlayback()",
            'case "response_started":',
            "type: \"audio\"",
            "type: \"text\"",
            "sessionStorage.getItem",
        ):
            with self.subTest(token=token):
                self.assertIn(token, html)
        self.assertNotIn("assistantSpeaking()) return;", html)

    def test_routes_are_registered(self) -> None:
        paths = {getattr(route, "path", None) for route in ui_server.app.routes}
        self.assertIn("/", paths)
        self.assertIn("/ws", paths)
        self.assertIn("/healthz", paths)
        self.assertIn("/menu/sync", paths)

    def test_prompt_requires_mock_order_tool(self) -> None:
        self.assertIn("MUST call", ui_server.DEFAULT_INSTRUCTIONS)
        self.assertIn("create_mock_order", ui_server.DEFAULT_INSTRUCTIONS)
        self.assertIn("MOCK_ORDER_CREATED", ui_server.DEFAULT_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
