"""Dependency-free PNG chart output with a strict reconciliation guard."""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Sequence

from services.reporting.contracts import PNG_CHART_SCHEMA_VERSION

_WIDTH, _HEIGHT, _MARGIN = 720, 360, 28


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def _pixel(canvas: bytearray, x: int, y: int, rgb: tuple[int, int, int]) -> None:
    if 0 <= x < _WIDTH and 0 <= y < _HEIGHT:
        offset = (y * _WIDTH + x) * 3
        canvas[offset : offset + 3] = bytes(rgb)


def _line(
    canvas: bytearray, x0: int, y0: int, x1: int, y1: int, rgb: tuple[int, int, int]
) -> None:
    dx, sx = abs(x1 - x0), (1 if x0 < x1 else -1)
    dy, sy = -abs(y1 - y0), (1 if y0 < y1 else -1)
    error = dx + dy
    while True:
        _pixel(canvas, x0, y0, rgb)
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _draw_axes(canvas: bytearray) -> tuple[int, int, int, int]:
    left, right = _MARGIN, _WIDTH - _MARGIN
    top, bottom = _MARGIN, _HEIGHT - _MARGIN
    for index in range(1, 5):
        y = top + index * (bottom - top) // 5
        _line(canvas, left, y, right, y, (225, 225, 225))
    _line(canvas, left, top, left, bottom, (80, 80, 80))
    _line(canvas, left, bottom, right, bottom, (80, 80, 80))
    return left, right, top, bottom


def _finite_values(values: Sequence[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        candidate = float(value)
        if math.isfinite(candidate):
            result.append(candidate)
    return result


def _series_points(
    values: list[float], bounds: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    left, right, top, bottom = bounds
    low, high = min(values), max(values)
    span = high - low if high > low else 1.0
    return [
        (
            left + index * (right - left) // (len(values) - 1),
            bottom - int((value - low) / span * (bottom - top)),
        )
        for index, value in enumerate(values)
    ]


def _draw_series(
    canvas: bytearray, values: list[float], bounds: tuple[int, int, int, int]
) -> None:
    if len(values) < 2:
        return
    points = _series_points(values, bounds)
    for first, second in zip(points, points[1:]):
        _line(canvas, first[0], first[1], second[0], second[1], (30, 95, 180))


def _encode_png(canvas: bytearray) -> bytes:
    scanlines = b"".join(
        b"\\x00" + bytes(canvas[row * _WIDTH * 3 : (row + 1) * _WIDTH * 3])
        for row in range(_HEIGHT)
    )
    header = struct.pack(">IIBBBBB", _WIDTH, _HEIGHT, 8, 2, 0, 0, 0)
    metadata = ("AliX-Chart-Schema\\x00" + PNG_CHART_SCHEMA_VERSION).encode("latin-1")
    return (
        b"\\x89PNG\\r\\n\\x1a\\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"tEXt", metadata)
        + _chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _chunk(b"IEND", b"")
    )


def render_equity_chart(
    values: Sequence[float],
    *,
    reconciliation_status: str,
    width: int = _WIDTH,
    height: int = _HEIGHT,
) -> bytes:
    """Plot an equity curve only for explicitly reconciled financial data."""
    if width != _WIDTH or height != _HEIGHT:
        raise ValueError("chart_dimensions_are_versioned_and_fixed")
    canvas = bytearray([255] * (_WIDTH * _HEIGHT * 3))
    bounds = _draw_axes(canvas)
    clean = _finite_values(values)
    if reconciliation_status == "RECONCILED":
        _draw_series(canvas, clean, bounds)
    return _encode_png(canvas)
