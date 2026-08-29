"""Chronological walk-forward split utilities."""

from __future__ import annotations

from collections.abc import Iterator


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    gap: int = 0,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Yield expanding-window train/test index tuples in chronological order.

    ``gap`` leaves observations between train and test, which is useful when a
    label uses a multi-day forward horizon or when a data release has a delay.
    """

    if n_samples < 1 or train_size < 1 or test_size < 1:
        raise ValueError("n_samples, train_size, and test_size must be positive")
    if gap < 0:
        raise ValueError("gap cannot be negative")
    step = test_size if step is None else step
    if step < 1:
        raise ValueError("step must be positive")

    train_end = train_size
    while train_end + gap < n_samples:
        test_start = train_end + gap
        test_end = min(test_start + test_size, n_samples)
        if test_start >= test_end:
            break
        yield tuple(range(train_end)), tuple(range(test_start, test_end))
        train_end += step

