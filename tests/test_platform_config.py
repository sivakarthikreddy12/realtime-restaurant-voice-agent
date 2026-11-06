"""Platform and model-profile selection tests."""

from __future__ import annotations

import unittest

from config.model_config import get_model_profile
from config.platform_config import detect_platform, resolve_platform


class PlatformConfigTests(unittest.TestCase):
    def test_platform_detection_normalizes_windows_and_arm64(self) -> None:
        self.assertTrue(detect_platform("Windows", "AMD64").is_windows)
        self.assertTrue(detect_platform("Darwin", "arm64").is_apple_silicon)
        self.assertFalse(detect_platform("Darwin", "x86_64").is_apple_silicon)

    def test_auto_profile_selection(self) -> None:
        self.assertEqual(resolve_platform("auto", system="Windows", machine="AMD64"), "windows-cpu")
        self.assertEqual(resolve_platform("auto", system="Darwin", machine="arm64"), "macos-apple-silicon")
        self.assertEqual(resolve_platform("auto", system="Linux", machine="x86_64"), "framework-defaults")

    def test_windows_profile_uses_cpu_compatible_models(self) -> None:
        args = get_model_profile("windows-cpu").cli_args()
        self.assertEqual(args[args.index("--stt") + 1], "faster-whisper")
        self.assertEqual(args[args.index("--faster_whisper_stt_compute_type") + 1], "int8")
        self.assertEqual(args[args.index("--model_name") + 1], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(args[args.index("--tts") + 1], "kokoro")
        self.assertNotIn("mlx-audio-whisper", args)

    def test_explicit_profiles_reject_the_wrong_platform(self) -> None:
        with self.assertRaises(ValueError):
            resolve_platform("windows-cpu", system="Linux", machine="x86_64")
        with self.assertRaises(ValueError):
            resolve_platform("macos-apple-silicon", system="Windows", machine="AMD64")


if __name__ == "__main__":
    unittest.main()
