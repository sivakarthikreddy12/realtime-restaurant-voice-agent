"""Bootstrap compatibility and memory reporting before the Realtime backend starts."""

from __future__ import annotations

import os
from warmup_compat import install_framework_compatibility


def main() -> None:
    profile = os.environ.get("RESTAURANT_VOICE_PROFILE", "framework-defaults")
    install_framework_compatibility(windows_cpu=profile == "windows-cpu")
    from speech_to_speech.cli import main as speech_to_speech_main

    speech_to_speech_main()


if __name__ == "__main__":
    main()
