"""Central runtime configuration for the restaurant voice agent."""

from .model_config import ModelProfile, get_model_profile
from .platform_config import RuntimePlatform, detect_platform, resolve_platform

__all__ = [
    "ModelProfile",
    "RuntimePlatform",
    "detect_platform",
    "get_model_profile",
    "resolve_platform",
]
