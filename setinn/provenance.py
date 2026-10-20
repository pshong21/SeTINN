"""What a run records about how it was produced: file hashes and the runtime."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Environment variables that affect JAX.
JAX_ENVIRONMENT = (
    "JAX_ENABLE_X64",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_FLAGS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON so that a reader never sees a half-written file."""

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def source_hashes() -> dict[str, str]:
    """sha256 of every file that can change a run: ``train.py`` and ``setinn/``."""

    files = [ROOT / "train.py", *sorted((ROOT / "setinn").rglob("*.py"))]
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in files
        if path.is_file() and "__pycache__" not in path.parts
    }


def implementation_hash(source_sha256: dict[str, str]) -> str:
    """One hash for the whole implementation."""

    payload = json.dumps(source_sha256, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def runtime_fingerprint(jax) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_environment": {
            name: os.environ[name] for name in JAX_ENVIRONMENT if name in os.environ
        },
        "devices": [
            {
                "id": getattr(device, "id", None),
                "platform": getattr(device, "platform", ""),
                "device_kind": getattr(device, "device_kind", str(device)),
            }
            for device in jax.devices()
        ],
    }
