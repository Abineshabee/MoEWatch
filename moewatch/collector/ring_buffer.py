# ------------------------------------------------------------------------------
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
# moewatch/collector/ring_buffer.py
#
# RingBuffer — fixed-capacity circular buffer for RoutingEvent objects.
#
# Why a ring buffer?
# ------------------
# Training runs can last millions of steps. A plain list would grow without
# bound, eventually OOM-ing the host or forcing an explicit flush. A ring
# buffer gives a hard memory ceiling: once full, the oldest events are
# silently overwritten. This is the correct trade-off for a diagnostic tool —
# we care about *recent* routing behaviour, not the full history.
#
# Design constraints
# ------------------
#   - O(1) append and O(n) snapshot — no shifting, no copying on write.
#   - Thread-safe for the single-writer / single-reader pattern that moewatch
#     uses (hook writes; analyzer reads). A threading.Lock guards the snapshot
#     path; the write path uses only an integer increment and modulo, which is
#     atomic enough under the GIL for this use case.
#   - Typed: stores only RoutingEvent instances. Rejects anything else with a
#     clear TypeError rather than silently corrupting the buffer.
#   - Zero torch dependency: the buffer is pure Python / numpy-free so it can
#     be imported and tested without a GPU environment.
#
# Author : Abinesh (GitHub: Abineshabee)
# License: Apache 2.0
# Version: 0.1.0
# ------------------------------------------------------------------------------

from __future__ import annotations

import logging
import threading
from typing import Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..hooks.router_hook import RoutingEvent  # circular-import guard

log = logging.getLogger(__name__)


class RingBuffer:
    """Fixed-capacity circular buffer for :class:`RoutingEvent` objects.

    Once the buffer reaches *capacity*, the oldest entry is overwritten.
    All reads return a consistent point-in-time snapshot — writes during
    iteration do not affect the returned list.

    Parameters
    ----------
    capacity : int
        Maximum number of events to retain. Must be ≥ 1.
        When the buffer wraps, a one-time DEBUG log is emitted.

    Examples
    --------
    >>> buf = RingBuffer(capacity=3)
    >>> for event in events:
    ...     buf.append(event)
    >>> recent = buf.snapshot()   # list of up to 3 most-recent events
    >>> buf.clear()
    """

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError(
                f"[moewatch] RingBuffer capacity must be a positive integer, "
                f"got {capacity!r}."
            )

        self._capacity: int = capacity
        self._buffer: List[Optional["RoutingEvent"]] = [None] * capacity
        self._head: int = 0  # next write position
        self._size: int = 0  # number of valid entries
        self._total_written: int = 0  # cumulative events written (never resets)
        self._wrapped: bool = False
        self._lock: threading.Lock = threading.Lock()

    # --------------------------------------------------------------------------
    # Write
    # --------------------------------------------------------------------------

    def append(self, event: "RoutingEvent") -> None:
        """Append *event* to the buffer, overwriting the oldest entry when full.

        Parameters
        ----------
        event : RoutingEvent
            Must be a ``RoutingEvent`` instance. A ``TypeError`` is raised for
            any other type to catch integration bugs early.

        Thread safety
        -------------
        Safe for single-writer use under the GIL. The write is not locked
        because the hook path must not stall the forward pass. The head
        pointer and size counter are updated with simple integer arithmetic
        that the GIL makes effectively atomic for this pattern.
        """
        # Import here to avoid circular import at module load time
        from ..hooks.router_hook import RoutingEvent

        if not isinstance(event, RoutingEvent):
            raise TypeError(
                f"[moewatch] RingBuffer.append() expects a RoutingEvent, "
                f"got {type(event).__name__}."
            )

        self._buffer[self._head] = event
        self._head = (self._head + 1) % self._capacity
        self._total_written += 1

        if self._size < self._capacity:
            self._size += 1
        elif not self._wrapped:
            self._wrapped = True
            log.debug(
                "[moewatch] RingBuffer full (capacity=%d). "
                "Oldest events will be overwritten. "
                "Increase ring_buffer_capacity in WatchConfig if you need "
                "longer history for trend analysis.",
                self._capacity,
            )

    # --------------------------------------------------------------------------
    # Read
    # --------------------------------------------------------------------------

    def snapshot(self) -> List["RoutingEvent"]:
        """Return a consistent copy of all current events, oldest-first.

        Acquires a lock so that a concurrent ``clear()`` cannot produce a
        torn read. Normal ``append()`` calls do not acquire this lock —
        a snapshot taken mid-write may miss the in-flight event, which is
        acceptable for a diagnostic tool.

        Returns
        -------
        list of RoutingEvent
            Ordered oldest → newest. Empty list if the buffer has never
            been written to.
        """
        with self._lock:
            if self._size == 0:
                return []

            if self._size < self._capacity:
                # Buffer has not yet wrapped — valid entries are 0 .. size-1
                return [e for e in self._buffer[: self._size] if e is not None]

            # Buffer has wrapped — oldest entry is at self._head
            tail = self._buffer[self._head :]
            head = self._buffer[: self._head]
            events = tail + head
            return [e for e in events if e is not None]

    def latest(self, n: int = 1) -> List["RoutingEvent"]:
        """Return the *n* most-recent events, newest-first.

        Parameters
        ----------
        n : int
            Number of events to return. Clamped to the current buffer size.
        """
        all_events = self.snapshot()
        return list(reversed(all_events[-n:])) if all_events else []

    def __iter__(self) -> Iterator["RoutingEvent"]:
        """Iterate over a snapshot (oldest → newest). Safe against concurrent writes."""
        return iter(self.snapshot())

    def __len__(self) -> int:
        """Number of valid events currently in the buffer (0 ≤ len ≤ capacity)."""
        return self._size

    # --------------------------------------------------------------------------
    # Reset
    # --------------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all events from the buffer without changing capacity.

        Acquires the lock so it is safe to call concurrently with snapshot().
        """
        with self._lock:
            self._buffer = [None] * self._capacity
            self._head = 0
            self._size = 0
            self._wrapped = False
        log.debug("[moewatch] RingBuffer cleared.")

    # --------------------------------------------------------------------------
    # Introspection
    # --------------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of events the buffer can hold."""
        return self._capacity

    @property
    def is_full(self) -> bool:
        """True when the buffer has reached capacity and is overwriting events."""
        return self._size == self._capacity

    @property
    def has_wrapped(self) -> bool:
        """True if at least one overwrite has occurred."""
        return self._wrapped

    @property
    def total_written(self) -> int:
        """Cumulative number of events written since creation (never resets)."""
        return self._total_written

    @property
    def utilization(self) -> float:
        """Fraction of capacity currently used (0.0 – 1.0)."""
        return self._size / self._capacity

    def __repr__(self) -> str:
        return (
            f"RingBuffer("
            f"size={self._size}/{self._capacity}, "
            f"wrapped={self._wrapped}, "
            f"total_written={self._total_written}"
            f")"
        )
