"""Small psutil-based RAM snapshots for the UI and backend processes."""

from __future__ import annotations

import os
from typing import Any


def memory_snapshot(root_pid: int | None = None) -> dict[str, int] | None:
    """Return physical RAM and resident memory for a process tree in MiB."""
    try:
        import psutil
    except ImportError:
        return None

    pid = root_pid or os.getpid()
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        process_rss = 0
        for process in processes:
            try:
                process_rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        virtual = psutil.virtual_memory()
        root_rss = root.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

    mib = 1024 * 1024
    return {
        "system_total_mb": int(virtual.total / mib),
        "system_available_mb": int(virtual.available / mib),
        "process_tree_rss_mb": int(process_rss / mib),
        "root_process_rss_mb": int(root_rss / mib),
    }


def log_memory_snapshot(stage: str, root_pid: int | None = None) -> dict[str, int] | None:
    snapshot = memory_snapshot(root_pid)
    if snapshot is None:
        print(f"[MEMORY] stage={stage} unavailable (psutil missing or process exited)", flush=True)
        return None
    details = " ".join(f"{key}={value}MiB" for key, value in snapshot.items())
    print(f"[MEMORY] stage={stage} {details}", flush=True)
    return snapshot


def memory_response(root_pid: int | None = None) -> dict[str, Any]:
    snapshot = memory_snapshot(root_pid)
    if snapshot is None:
        return {"status": "unavailable", "reason": "psutil is missing or the process exited"}
    return {"status": "ok", **snapshot}
