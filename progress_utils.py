from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

try:  # pragma: no cover
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


@dataclass
class _NullProgressBar:
    total: Optional[int] = None
    desc: Optional[str] = None
    unit: str = "it"
    leave: bool = True

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "_NullProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return None


def progress_bar(*, total: Optional[int] = None, desc: Optional[str] = None, unit: str = "it", leave: bool = True):
    if _tqdm is None:
        return _NullProgressBar(total=total, desc=desc, unit=unit, leave=leave)
    return _tqdm(total=total, desc=desc, unit=unit, leave=leave)


def tqdm_iter(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: str = "it",
    leave: bool = True,
):
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave)
