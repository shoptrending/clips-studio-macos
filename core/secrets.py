"""Somewhere to keep OAuth tokens that isn't a plaintext file.

A 0600 file protects against another user on the box, not against someone
who already has your account. Nothing here ever logs a value.
"""

import json
import os
from pathlib import Path

_DIR = "credentials"


def backend_name() -> str:
    return "file"


def _path(data_dir: Path, name: str) -> Path:
    return Path(data_dir) / _DIR / f"{name}.secret.json"


def save(data_dir: Path, name: str, payload: dict) -> None:
    target = _path(data_dir, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload).encode("utf-8")
    target.write_bytes(raw)
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass


def load(data_dir: Path, name: str) -> dict | None:
    """Read a stored secret, or None if it isn't there or can't be read.

    Never raises.
    """
    try:
        raw = _path(data_dir, name).read_bytes()
        value = json.loads(raw.decode("utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def wipe(data_dir: Path, name: str) -> bool:
    """Delete a stored secret. True if it is gone."""
    try:
        _path(data_dir, name).unlink(missing_ok=True)
        return True
    except OSError:
        print("A stored credential could not be deleted and is still on disk.")
        return False


def has(data_dir: Path, name: str) -> bool:
    return _path(data_dir, name).exists()


def migrate_plaintext(data_dir: Path, name: str, legacy: Path) -> bool:
    """Move an old plaintext credential file into the store and delete it."""
    if has(data_dir, name) or not legacy.exists():
        return False
    try:
        payload = json.loads(legacy.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict) or not payload:
        return False
    save(data_dir, name, payload)
    try:
        legacy.unlink()
    except OSError:
        print(f"Moved {legacy.name} into the credential store, but could not delete the original.")
        return True
    print(f"Moved {legacy.name} into the credential store.")
    return True
