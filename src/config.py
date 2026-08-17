"""Tiny YAML config loader with dotted-key CLI overrides."""

from __future__ import annotations

from typing import Any, List

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce(v: str) -> Any:
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if v.lower() in {"none", "null"}:
        return None
    return v


def apply_overrides(cfg: dict, overrides: List[str]) -> dict:
    """Apply ``a.b.c=value`` overrides in place (values are type-coerced)."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override '{ov}' must be key=value")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg
