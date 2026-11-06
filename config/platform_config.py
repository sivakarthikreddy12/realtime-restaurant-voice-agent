"""Platform detection and named runtime profile selection."""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePlatform:
    system: str
    machine: str

    @property
    def is_windows(self) -> bool:
        return self.system == "windows"

    @property
    def is_apple_silicon(self) -> bool:
        return self.system == "darwin" and self.machine in {"arm64", "aarch64"}


def detect_platform(system: str | None = None, machine: str | None = None) -> RuntimePlatform:
    """Return normalized platform information, with injectable values for tests."""
    raw_system = (system or platform.system()).strip().lower()
    raw_machine = (machine or platform.machine()).strip().lower()
    if raw_system.startswith("win"):
        normalized_system = "windows"
    elif raw_system in {"darwin", "mac", "macos"}:
        normalized_system = "darwin"
    elif raw_system.startswith("linux"):
        normalized_system = "linux"
    else:
        normalized_system = raw_system
    return RuntimePlatform(normalized_system, raw_machine)


def resolve_platform(
    profile: str = "auto",
    *,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    """Resolve a named profile or infer one from the current operating system."""
    current = detect_platform(system, machine)
    normalized = profile.strip().lower().replace("_", "-")
    if normalized == "auto":
        if current.is_windows:
            return "windows-cpu"
        if current.is_apple_silicon:
            return "macos-apple-silicon"
        return "framework-defaults"
    if normalized == "windows-cpu":
        if not current.is_windows:
            raise ValueError("The windows-cpu profile can only run on Windows.")
        return normalized
    if normalized == "macos-apple-silicon":
        if not current.is_apple_silicon:
            raise ValueError("The macos-apple-silicon profile requires Apple Silicon macOS.")
        return normalized
    if normalized == "framework-defaults":
        return normalized
    raise ValueError(
        "Unknown runtime profile. Choose auto, windows-cpu, macos-apple-silicon, or framework-defaults."
    )
