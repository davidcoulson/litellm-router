"""Minimal Prometheus text-format parser (no external requirements)."""

from __future__ import annotations

import math
from dataclasses import dataclass

_ESCAPES = {"\\": "\\", '"': '"', "n": "\n"}


@dataclass(frozen=True, slots=True)
class Sample:
    """One sample line."""

    name: str
    labels: dict[str, str]
    value: float


def _parse_labels(line: str, start: int) -> tuple[dict[str, str], int] | None:
    """Parse labels beginning just after '{'. Return (labels, index after '}')."""
    labels: dict[str, str] = {}
    i, n = start, len(line)
    while i < n:
        while i < n and line[i] in " ,":
            i += 1
        if i < n and line[i] == "}":
            return labels, i + 1
        eq = line.find("=", i)
        if eq == -1 or eq + 1 >= n or line[eq + 1] != '"':
            return None
        key = line[i:eq].strip()
        i = eq + 2
        chars: list[str] = []
        while i < n and line[i] != '"':
            if line[i] == "\\" and i + 1 < n:
                chars.append(_ESCAPES.get(line[i + 1], line[i + 1]))
                i += 2
            else:
                chars.append(line[i])
                i += 1
        if i >= n:
            return None
        labels[key] = "".join(chars)
        i += 1
    return None


def _parse_line(line: str) -> Sample | None:
    brace = line.find("{")
    if brace == -1:
        parts = line.split()
        if len(parts) < 2:
            return None
        name, labels, rest = parts[0], {}, parts[1:]
    else:
        name = line[:brace].strip()
        parsed = _parse_labels(line, brace + 1)
        if parsed is None:
            return None
        labels, end = parsed
        rest = line[end:].split()
    if not name or not rest:
        return None
    try:
        value = float(rest[0])
    except ValueError:
        return None
    if math.isnan(value):
        return None
    return Sample(name, labels, value)


def parse_prometheus(text: str) -> list[Sample]:
    """Parse exposition text into samples, skipping comments, NaN and bad lines."""
    samples: list[Sample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sample = _parse_line(line)
        if sample is not None:
            samples.append(sample)
    return samples
