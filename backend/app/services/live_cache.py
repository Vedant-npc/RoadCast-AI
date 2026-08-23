from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int = 60) -> None:
        self.ttl_seconds = ttl_seconds
        self._entry: _CacheEntry[T] | None = None

    def get(self) -> T | None:
        if self._entry is None or monotonic() >= self._entry.expires_at:
            self._entry = None
            return None
        return self._entry.value

    def set(self, value: T) -> T:
        self._entry = _CacheEntry(value, monotonic() + self.ttl_seconds)
        return value

    def clear(self) -> None:
        self._entry = None
