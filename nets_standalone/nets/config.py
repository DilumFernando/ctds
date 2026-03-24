from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable
import tomllib


class Config(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _apply_override(cfg: Config, override: str) -> None:
    key, raw_value = override.split("=", 1)
    value = _wrap(_parse_value(raw_value))
    target = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], Config):
            target[part] = Config()
        target = target[part]
    target[parts[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    with open(Path(path), "rb") as handle:
        cfg = _wrap(tomllib.load(handle))
    for override in overrides or []:
        _apply_override(cfg, override)
    return cfg
