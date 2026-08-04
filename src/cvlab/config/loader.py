"""Generic dataclass <- YAML loading, shared by every config type in `schema.py`.

Kept dependency-free (no dacite/pydantic) per the "keep dependencies minimal"
constraint: this module is the one place that knows how to turn a nested dict
into a nested dataclass.
"""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

from cvlab.config.schema import RunConfig

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a plain dict."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a YAML mapping at the top level, got {type(data)}")
    return data


def load_dataclass(cls: type[T], data: dict[str, Any]) -> T:
    """Recursively construct a dataclass instance from a (possibly nested) dict.

    Unknown keys in `data` are ignored; missing keys fall back to the
    dataclass's own defaults. Dataclass-typed fields (direct or inside
    Optional[...]) are recursed into when the value is itself a dict.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")

    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints[f.name], data[f.name])
    return cls(**kwargs)


def _coerce(field_type: Any, value: Any) -> Any:
    origin = typing.get_origin(field_type)

    if origin is typing.Union:
        args = [a for a in typing.get_args(field_type) if a is not type(None)]
        if value is None:
            return None
        for candidate in args:
            if dataclasses.is_dataclass(candidate) and isinstance(value, dict):
                return load_dataclass(candidate, value)
        return value

    if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
        return load_dataclass(field_type, value)

    return value


def load_run_config(path: str | Path) -> RunConfig:
    """Load a single, fully-merged YAML file into a `RunConfig`.

    Composing the separate `configs/datasets/*.yaml`, `configs/encoders/*.yaml`,
    and `configs/methods/*.yaml` fragments into one of these is the sweep
    runner's job (M8); this function just does the dict -> dataclass step for
    an already-merged mapping, e.g. in tests or one-off runs.
    """
    return load_dataclass(RunConfig, load_yaml(path))
