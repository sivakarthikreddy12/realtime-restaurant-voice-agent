"""Tests for launcher path setup, TLS fallback, and Realtime session wiring."""

from __future__ import annotations

import json
import os
import ssl
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as launcher  # noqa: E402
import order_tools  # noqa: E402


class LauncherTests(unittest.TestCase):
    def test_project_root_is_added_to_sys_path(self) -> None:
        self.assertIn(str(ROOT), sys.path)

    def test_build_s2s_argv_has_tool_and_english_defaults(self) -> None:
        options = launcher.LauncherOptions()
        argv = launcher.build_s2s_argv(options)
        self.assertEqual(argv[0], "local")
        self.assertIn("--tool-module", argv)
        self.assertEqual(argv[argv.index("--tool-module") + 1], "order_tools")
        self.assertIn("--init_chat_prompt", argv)
        self.assertIn("--qwen3_tts_language", argv)
        self.assertEqual(argv[argv.index("--qwen3_tts_language") + 1], "en")
        self.assertIn("--port", argv)

    def test_mac_optimal_flag_only_on_apple_silicon(self) -> None:
        with patch.object(launcher, "_on_apple_silicon", return_value=True):
            self.assertIn("--mac-optimal-settings", launcher.build_s2s_argv(launcher.LauncherOptions()))
        with patch.object(launcher, "_on_apple_silicon", return_value=False):
            self.assertNotIn(
                "--mac-optimal-settings", launcher.build_s2s_argv(launcher.LauncherOptions())
            )
        self.assertNotIn(
            "--mac-optimal-settings",
            launcher.build_s2s_argv(launcher.LauncherOptions(mac_optimal=False)),
        )

    def test_extra_args_are_forwarded(self) -> None:
        options = launcher.parse_args(["--", "--verbose", "--local_audio_print_json"])
        self.assertEqual(options.extra_args, ["--verbose", "--local_audio_print_json"])
        argv = launcher.build_s2s_argv(options)
        self.assertEqual(argv[-2:], ["--verbose", "--local_audio_print_json"])

    def test_ssl_fallback_uses_certifi_with_verification_enabled(self) -> None:
        broken = {
            "SSL_CERT_FILE": "/tmp/definitely-missing-ca.pem",
            "SSL_CERT_DIR": "/tmp/definitely-missing-ca-dir",
        }
        with patch.dict(os.environ, broken, clear=False):
            # The broken paths must be detected as unusable.
            self.assertFalse(launcher._default_ssl_context_has_certs())
            bundle = launcher.configure_ssl_trust()
            self.assertIsNotNone(bundle)
            self.assertEqual(os.environ["SSL_CERT_FILE"], bundle)

            context = ssl.create_default_context(cafile=bundle)
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
            self.assertGreater(len(context.get_ca_certs()), 0)


class RealtimeContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        order_tools.reset_state()

    async def test_client_loader_executes_tool_module(self) -> None:
        from speech_to_speech.api.openai_realtime.audio_client import load_realtime_tool_module

        tools, executor, create_response = load_realtime_tool_module("order_tools")
        self.assertEqual(len(tools), 5)
        self.assertIs(create_response, True)
        result = await executor("add_to_cart", {"item_name": "Coke", "quantity": 2})
        payload = json.loads(json.dumps(result))
        self.assertEqual(payload["subtotal"], 3.98)

    async def test_tool_call_coordinator_serializes_results(self) -> None:
        from speech_to_speech.api.openai_realtime.audio_client import (
            RealtimeAudioClientConfig,
            _ToolCallCoordinator,
            load_realtime_tool_module,
        )

        tools, executor, create_response = load_realtime_tool_module("order_tools")
        config = RealtimeAudioClientConfig(
            tools=tools,
            tool_executor=executor,
            tool_response_create=create_response,
        )
        coordinator = _ToolCallCoordinator(conn=None, config=config)

        execution = await coordinator._execute_tool(
            "call-1", "add_to_cart", json.dumps({"item_name": "Coke", "quantity": 2})
        )
        self.assertTrue(execution.create_response)
        payload = json.loads(execution.output)
        self.assertEqual(payload["added"], {"name": "Coke", "quantity": 2})
        self.assertEqual(payload["subtotal"], 3.98)

        execution = await coordinator._execute_tool(
            "call-2", "get_cart", json.dumps({})
        )
        self.assertTrue(json.loads(execution.output)["ready_for_confirmation"])

        execution = await coordinator._execute_tool(
            "call-3",
            "create_mock_order",
            json.dumps({"customer_confirmation": "yes, place the order"}),
        )
        receipt = json.loads(execution.output)
        self.assertEqual(receipt["status"], "MOCK_ORDER_CREATED")
        self.assertFalse(receipt["real_order_placed"])

    async def test_session_update_includes_tools_and_audio_rates(self) -> None:
        from speech_to_speech.api.openai_realtime.audio_client import (
            RealtimeAudioClientConfig,
            build_session_update,
            load_realtime_tool_module,
        )
        from ui_server import build_browser_session_update

        tools, executor, create_response = load_realtime_tool_module("order_tools")
        config = RealtimeAudioClientConfig(
            tools=tools,
            tool_executor=executor,
            tool_response_create=create_response,
            send_rate=16000,
            recv_rate=16000,
        )
        session_update = build_session_update(config)
        session = session_update["session"]
        self.assertEqual(session["type"], "realtime")
        self.assertEqual(session["tool_choice"], "auto")
        self.assertEqual(
            [tool["name"] for tool in session["tools"]],
            ["show_menu", "add_to_cart", "remove_from_cart", "get_cart", "create_mock_order"],
        )
        self.assertEqual(session["audio"]["input"]["turn_detection"]["type"], "server_vad")
        browser_detection = build_browser_session_update(config)["session"]["audio"]["input"]["turn_detection"]
        self.assertTrue(browser_detection["create_response"])
        self.assertEqual(browser_detection["silence_duration_ms"], 700)
        # At the pipeline's native 16 kHz the client omits an explicit format.
        self.assertNotIn("format", session["audio"]["input"])
        self.assertFalse(session["audio"]["input"].get("format"))

    async def test_invalid_tool_arguments_are_reported_not_raised(self) -> None:
        from speech_to_speech.api.openai_realtime.audio_client import (
            RealtimeAudioClientConfig,
            _ToolCallCoordinator,
            load_realtime_tool_module,
        )

        tools, executor, create_response = load_realtime_tool_module("order_tools")
        config = RealtimeAudioClientConfig(
            tools=tools,
            tool_executor=executor,
            tool_response_create=create_response,
        )
        coordinator = _ToolCallCoordinator(conn=None, config=config)
        execution = await coordinator._execute_tool(
            "call-bad", "add_to_cart", json.dumps({"item_name": "Coke", "quantity": 0})
        )
        self.assertTrue(execution.create_response)
        self.assertIn("error", json.loads(execution.output))


if __name__ == "__main__":
    unittest.main()
