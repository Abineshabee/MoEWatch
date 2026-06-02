# ------------------------------------------------------------------------------
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
# moewatch/collector/stat_collector.py
#
# StatCollector — aggregates RoutingEvents from all router layers into
# per-layer statistics that the analyzer layer can consume.
#
# LayerStats — typed dict-like container for one layer's aggregated stats.
#
# Architecture
# ------------
#
#   RouterHook (per layer)
#       │  writes RoutingEvent via add_event()
#       ▼
#   StatCollector
#       │  maintains one RingBuffer per layer
#       │  accumulates running totals (expert_counts, total_tokens)
#       │  computes load_imbalance_score on demand
#       ▼
#   get_all_stats() → Dict[layer_name, LayerStats]
#       │
#       ▼
#   EntropyAnalyzer / CollapseDetector (analyzer layer)
#
# Design constraints
# ------------------
#   - CPU-only: all tensor operations use .cpu() data. No GPU sync calls.
#   - Streaming accumulation: running totals are updated on every add_event()
#     so that get_all_stats() is O(layers), not O(events).
#   - Window-aware: the rolling window (config.window_steps) is applied by
#     filtering the ring buffer snapshot rather than maintaining a second
#     data structure. This is O(window_steps) on read but O(1) on write,
#     which is the right trade-off for a diagnostic tool where reads are rare.
#   - Thread-safe for the single-writer (hook) / single-reader (analyzer)
#     pattern. Concurrent writers are not supported.
#
# Author : Abinesh (GitHub: Abineshabee)
# License: Apache 2.0
# Version: 0.1.0
# ------------------------------------------------------------------------------

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

import torch

from ..config import WatchConfig
from .ring_buffer import RingBuffer

if TYPE_CHECKING:
    from ..hooks.router_hook import RoutingEvent

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# LayerStats — per-layer aggregated statistics
# ------------------------------------------------------------------------------

@dataclass
class LayerStats:
    """Aggregated routing statistics for a single MoE router layer.

    Produced by :meth:`StatCollector.get_layer_stats` and consumed by the
    analyzer layer (:class:`~moewatch.analyzer.entropy.EntropyAnalyzer` and
    :class:`~moewatch.analyzer.collapse.CollapseDetector`).

    Attributes
    ----------
    layer_name : str
        Fully-qualified module name.
    n_experts : int
        Total number of experts in this layer. ``0`` if no events have been
        collected yet.
    expert_counts : torch.Tensor
        1-D int64 CPU tensor of shape ``(n_experts,)`` representing the total
        number of tokens routed to each expert across all collected events
        within the rolling window.
    total_tokens : int
        Sum of ``expert_counts`` — total tokens processed across all sampled
        steps in the window.
    utilization : torch.Tensor
        Float32 tensor of shape ``(n_experts,)`` — fraction of tokens sent
        to each expert (``expert_counts / total_tokens``). All-zeros when
        ``total_tokens == 0``.
    load_imbalance_score : float
        Ratio of the most-loaded expert to the mean load. Perfect balance → 1.0.
        Higher values indicate more skewed routing. ``float('nan')`` when
        ``total_tokens == 0``.
    top_k : int
        Number of experts each token was routed to (inferred from events).
    event_count : int
        Number of ``RoutingEvent`` objects that contributed to these stats.
    step_range : tuple of (int, int)
        ``(first_step, last_step)`` of the contributing events.
        ``(-1, -1)`` when no events have been collected.
    has_raw_logits : bool
        True if at least one contributing event contained raw logit tensors.
        When True, entropy can be computed from logits; when False, entropy
        is estimated from token counts alone.
    raw_logits_window : list of torch.Tensor, optional
        Raw logit tensors from the most-recent events (up to window_steps).
        Each tensor has shape ``(total_tokens_in_batch, n_experts)``.
        ``None`` when no raw logits are available.
    """

    layer_name:           str
    n_experts:            int
    expert_counts:        torch.Tensor          # shape: (n_experts,), int64, CPU
    total_tokens:         int
    utilization:          torch.Tensor          # shape: (n_experts,), float32, CPU
    load_imbalance_score: float
    top_k:                int                   = 1
    event_count:          int                   = 0
    step_range:           tuple                 = (-1, -1)
    has_raw_logits:       bool                  = False
    raw_logits_window:    Optional[List[torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        """True when no events have been collected for this layer."""
        return self.event_count == 0


    def dead_mask(self, threshold: float = 0.001) -> torch.Tensor:
        """Boolean mask — True for experts at or below *threshold* utilisation."""
        return self.utilization <= threshold

    def to_dict(self) -> dict:
        """JSON-serialisable summary (tensors converted to Python lists)."""
        return {
            "layer_name":           self.layer_name,
            "n_experts":            self.n_experts,
            "expert_counts":        self.expert_counts.tolist(),
            "total_tokens":         self.total_tokens,
            "utilization":          self.utilization.tolist(),
            "load_imbalance_score": (
                self.load_imbalance_score
                if self.load_imbalance_score == self.load_imbalance_score  # NaN check
                else None
            ),
            "top_k":                self.top_k,
            "event_count":          self.event_count,
            "step_range":           list(self.step_range),
            "has_raw_logits":       self.has_raw_logits,
        }


# ------------------------------------------------------------------------------
# _LayerAccumulator — internal per-layer mutable state
# ------------------------------------------------------------------------------

@dataclass
class _LayerAccumulator:
    """Internal mutable accumulator for one router layer.

    All public data is exposed through StatCollector; this class is
    never part of the public API.
    """

    layer_name:      str
    ring_buffer:     RingBuffer
    # Running totals — updated on every add_event() call
    running_counts:  Optional[torch.Tensor]  = None   # (n_experts,) int64
    total_tokens:    int                     = 0
    n_experts:       int                     = 0
    top_k:           int                     = 1
    first_step:      int                     = -1
    last_step:       int                     = -1
    event_count:     int                     = 0

    def update(self, event: "RoutingEvent") -> None:
        """Integrate one RoutingEvent into the running totals."""
        if event.n_experts == 0:
            # Zero-count event from fallback extraction — skip accumulation
            log.debug(
                "[moewatch] StatCollector: skipping zero-count event for layer %s",
                self.layer_name,
            )
            return

        # First event — initialise accumulators from the event dimensions
        if self.running_counts is None:
            self.n_experts      = event.n_experts
            self.running_counts = torch.zeros(event.n_experts, dtype=torch.int64)

        # Handle n_experts mismatch (architecture change mid-run — rare but possible)
        elif event.n_experts != self.n_experts:
            log.warning(
                "[moewatch] StatCollector(%s): n_experts changed from %d to %d. "
                "Resetting running totals. This may indicate a model with "
                "heterogeneous MoE layers or a hook mis-assignment.",
                self.layer_name,
                self.n_experts,
                event.n_experts,
            )
            self.n_experts      = event.n_experts
            self.running_counts = torch.zeros(event.n_experts, dtype=torch.int64)
            self.total_tokens   = 0

        self.running_counts += event.expert_counts.to(torch.int64)
        self.total_tokens   += event.total_tokens
        self.top_k           = event.top_k
        self.event_count    += 1

        if self.first_step == -1:
            self.first_step = event.step
        self.last_step = event.step


# ------------------------------------------------------------------------------
# StatCollector
# ------------------------------------------------------------------------------

class StatCollector:
    """Aggregates :class:`~moewatch.hooks.router_hook.RoutingEvent` objects
    from all instrumented router layers into per-layer statistics.

    One ``StatCollector`` is shared by all ``RouterHook`` instances attached
    to the same model. It maintains one :class:`RingBuffer` per layer plus
    running accumulator totals for O(1) write and O(layers) read.

    Parameters
    ----------
    layer_names : list of str
        Fully-qualified names of all router modules to track. Layers not
        in this list are silently ignored if ``add_event()`` is called for
        them (defensive: the hook might fire before the collector is aware).
    config : WatchConfig
        Provides ``ring_buffer_capacity`` and ``window_steps``.

    Examples
    --------
    >>> collector = StatCollector(layer_names=["model.layers.0.moe"], config=cfg)
    >>> collector.add_event(event)          # called by RouterHook
    >>> stats = collector.get_all_stats()   # called by analyzers
    >>> stats["model.layers.0.moe"].utilization
    tensor([0.1234, 0.0900, ...])
    """

    def __init__(
        self,
        layer_names: List[str],
        config: WatchConfig,
    ) -> None:
        self.config = config

        self._accumulators: Dict[str, _LayerAccumulator] = {
            name: _LayerAccumulator(
                layer_name=name,
                ring_buffer=RingBuffer(capacity=config.ring_buffer_capacity),
            )
            for name in layer_names
        }

        # Lock used only for get_all_stats() / clear() — not for add_event()
        self._read_lock = threading.Lock()

        log.debug(
            "[moewatch] StatCollector initialised for %d layer(s).",
            len(layer_names),
        )

    # --------------------------------------------------------------------------
    # Write path (called by RouterHook — must be fast)
    # --------------------------------------------------------------------------

    def add_event(self, event: "RoutingEvent") -> None:
        """Integrate a single :class:`RoutingEvent` into the collector.

        Called directly from :class:`~moewatch.hooks.router_hook.RouterHook`
        on every sampled forward pass. Must never raise — any error is caught
        and logged so the training step continues unaffected.

        Parameters
        ----------
        event : RoutingEvent
            The event to integrate. If the event's ``layer_name`` is not in
            the collector's tracked layers, a new accumulator is created on
            the fly (handles cases where detection runs after the hook fires).
        """
        try:
            acc = self._accumulators.get(event.layer_name)
            if acc is None:
                # Defensive: auto-register unknown layers rather than dropping data
                log.debug(
                    "[moewatch] StatCollector: auto-registering untracked layer %s.",
                    event.layer_name,
                )
                acc = _LayerAccumulator(
                    layer_name=event.layer_name,
                    ring_buffer=RingBuffer(capacity=self.config.ring_buffer_capacity),
                )
                self._accumulators[event.layer_name] = acc

            acc.ring_buffer.append(event)
            acc.update(event)

        except Exception as exc:                    # pragma: no cover
            log.warning(
                "[moewatch] StatCollector.add_event() error for layer %s: %s",
                getattr(event, "layer_name", "unknown"),
                exc,
            )

    # --------------------------------------------------------------------------
    # Read path (called by analyzers — O(layers × window_steps))
    # --------------------------------------------------------------------------

    def get_layer_stats(self, layer_name: str) -> Optional[LayerStats]:
        """Return aggregated :class:`LayerStats` for a single layer.

        Statistics are computed over the most-recent ``config.window_steps``
        events (rolling window), not the full history. This ensures that
        collapse detection reflects current routing behaviour rather than
        being diluted by healthy early training steps.

        Parameters
        ----------
        layer_name : str
            The layer to query. Returns ``None`` if the layer is unknown.
        """
        with self._read_lock:
            acc = self._accumulators.get(layer_name)
            if acc is None:
                return None
            return self._compute_stats(acc)

    def get_all_stats(self) -> Dict[str, LayerStats]:
        """Return a snapshot of :class:`LayerStats` for every tracked layer.

        Layers with no events (empty ring buffer) are included with zero
        counts so that the analyzer can report on them explicitly.

        Returns
        -------
        dict
            ``{layer_name: LayerStats}`` — ordered by layer registration order.
        """
        with self._read_lock:
            return {
                name: self._compute_stats(acc)
                for name, acc in self._accumulators.items()
            }

    def _compute_stats(self, acc: _LayerAccumulator) -> LayerStats:
        """Build a :class:`LayerStats` from *acc* using the rolling window.

        This is the only place where ring buffer snapshots are taken and
        windowed tensors are computed. Everything else uses running totals.
        """
        # -- Empty accumulator (no events yet) --------------------------------
        if acc.running_counts is None or acc.event_count == 0:
            dummy = torch.zeros(0, dtype=torch.int64)
            return LayerStats(
                layer_name=acc.layer_name,
                n_experts=0,
                expert_counts=dummy,
                total_tokens=0,
                utilization=dummy.float(),
                load_imbalance_score=float("nan"),
                event_count=0,
                step_range=(-1, -1),
            )

        # -- Rolling window: filter ring buffer to last window_steps events ---
        all_events = acc.ring_buffer.snapshot()
        window_events = all_events[-self.config.window_steps:]

        if not window_events:
            # Ring buffer cleared between lock acquisition — return running totals
            window_events = all_events

        # -- Aggregate counts over the window ---------------------------------
        n_experts     = acc.n_experts
        window_counts = torch.zeros(n_experts, dtype=torch.int64)
        window_tokens = 0
        raw_logits_list: List[torch.Tensor] = []
        has_raw_logits  = False

        for ev in window_events:
            if ev.n_experts != n_experts:
                # Skip mismatched events (architecture change edge case)
                continue
            window_counts += ev.expert_counts.to(torch.int64)
            window_tokens += ev.total_tokens

            if ev.raw_logits is not None:
                raw_logits_list.append(ev.raw_logits)
                has_raw_logits = True

        # -- Utilisation ------------------------------------------------------
        if window_tokens > 0:
            utilization = window_counts.float() / window_tokens
        else:
            utilization = torch.zeros(n_experts, dtype=torch.float32)

        # -- Load imbalance score (max / mean) --------------------------------
        if window_tokens > 0 and n_experts > 0:
            mean_util = utilization.mean().item()
            max_util  = utilization.max().item()
            load_imbalance = max_util / mean_util if mean_util > 0 else float("nan")
        else:
            load_imbalance = float("nan")

        # -- Step range -------------------------------------------------------
        if window_events:
            first_step = window_events[0].step
            last_step  = window_events[-1].step
        else:
            first_step = acc.first_step
            last_step  = acc.last_step

        return LayerStats(
            layer_name=acc.layer_name,
            n_experts=n_experts,
            expert_counts=window_counts,
            total_tokens=window_tokens,
            utilization=utilization,
            load_imbalance_score=load_imbalance,
            top_k=acc.top_k,
            event_count=len(window_events),
            step_range=(first_step, last_step),
            has_raw_logits=has_raw_logits,
            raw_logits_window=raw_logits_list if has_raw_logits else None,
        )

    # --------------------------------------------------------------------------
    # Maintenance
    # --------------------------------------------------------------------------

    def clear(self, layer_name: Optional[str] = None) -> None:
        """Clear collected events.

        Parameters
        ----------
        layer_name : str, optional
            If provided, clear only that layer's buffer and running totals.
            If ``None``, clear all layers.
        """
        with self._read_lock:
            targets = (
                [self._accumulators[layer_name]]
                if layer_name and layer_name in self._accumulators
                else list(self._accumulators.values())
            )
            for acc in targets:
                acc.ring_buffer.clear()
                acc.running_counts = None
                acc.total_tokens   = 0
                acc.event_count    = 0
                acc.first_step     = -1
                acc.last_step      = -1

        scope = layer_name or "all layers"
        log.debug("[moewatch] StatCollector cleared: %s.", scope)

    # --------------------------------------------------------------------------
    # Introspection
    # --------------------------------------------------------------------------

    @property
    def tracked_layers(self) -> List[str]:
        """Names of all tracked layers in registration order."""
        return list(self._accumulators.keys())

    @property
    def total_events(self) -> int:
        """Total events written across all layers (sum of ring buffer totals)."""
        return sum(
            acc.ring_buffer.total_written
            for acc in self._accumulators.values()
        )

    def buffer_utilization(self) -> Dict[str, float]:
        """Return the ring buffer fill fraction for each layer (0.0 – 1.0)."""
        return {
            name: acc.ring_buffer.utilization
            for name, acc in self._accumulators.items()
        }

    def summary(self) -> str:
        """Return a one-line text summary of collector state."""
        n_layers     = len(self._accumulators)
        total_events = self.total_events
        full_buffers = sum(
            1 for acc in self._accumulators.values()
            if acc.ring_buffer.is_full
        )
        return (
            f"StatCollector: {n_layers} layer(s), "
            f"{total_events} total event(s), "
            f"{full_buffers}/{n_layers} buffer(s) full"
        )

    def __repr__(self) -> str:
        return (
            f"StatCollector("
            f"layers={len(self._accumulators)}, "
            f"total_events={self.total_events}, "
            f"window_steps={self.config.window_steps}"
            f")"
        )
