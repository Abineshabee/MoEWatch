# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  analyzer/collapse.py — Expert collapse detection for MoE routing diagnostics
#
#  Expert collapse is the primary failure mode in Mixture-of-Experts training.
#  It occurs when the router consistently routes most tokens to a small subset
#  of experts, leaving others permanently inactive. Once collapsed, the model
#  wastes parameter capacity and typically cannot recover without explicit
#  intervention (auxiliary loss adjustment, re-initialisation).
#
#  This module distinguishes three expert health states:
#
#    HEALTHY  — Expert is consistently receiving tokens above cold_threshold.
#               No action required.
#
#    COLD     — Expert utilisation has dropped below cold_threshold but is
#               above dead_threshold. The expert may recover; this is an
#               early warning. Promoted to DEAD after cold_steps_limit
#               consecutive cold observations.
#
#    DEAD     — Expert utilisation is at or below dead_threshold, or the
#               expert has been COLD for more than cold_steps_limit steps.
#               Capacity is permanently wasted; intervention is recommended.
#
#  The distinction between COLD and DEAD is critical for useful diagnostics:
#  flagging every temporarily under-utilised expert as dead would produce
#  alert fatigue during early training when the router is still learning.
#
#  Public API
#  ----------
#    ExpertState         — Enum: HEALTHY | COLD | DEAD
#    ExpertStatus        — Immutable snapshot for one expert: state, utilisation,
#                          consecutive cold steps, alert level.
#    LayerCollapseReport — All ExpertStatus objects for one layer plus summary
#                          metrics (n_dead, n_cold, load_imbalance_score).
#    CollapseDetector    — Stateful analyser. Consumes Dict[str, LayerStats]
#                          and emits a Dict[str, LayerCollapseReport].
#
#  State tracking
#  --------------
#  CollapseDetector maintains a per-layer, per-expert "consecutive cold step"
#  counter (``_cold_counters``). This is necessary because LayerStats provides
#  only a rolling-window snapshot — the detector must remember whether an
#  expert was cold in the previous call to increment the counter correctly.
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#  Version: 0.1.0
#
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch

from ..config import AlertLevel, WatchConfig
from ..collector.stat_collector import LayerStats

log = logging.getLogger(__name__)


# =============================================================================
# Section 1 — Expert state classification
# =============================================================================


class ExpertState(str, Enum):
    """Health state for a single MoE expert.

    Attributes
    ----------
    HEALTHY:
        Expert is actively receiving tokens above ``cold_threshold``.
    COLD:
        Expert utilisation has dropped below ``cold_threshold`` but remains
        above ``dead_threshold``. May recover; triggers a WARN alert.
    DEAD:
        Expert is at or below ``dead_threshold``, or has been COLD for more
        than ``cold_steps_limit`` steps. Triggers an ERROR alert.
    UNKNOWN:
        No routing data has been collected for this layer yet.
    """

    HEALTHY = "HEALTHY"
    COLD = "COLD"
    DEAD = "DEAD"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# Section 2 — Data containers
# =============================================================================

# ------------------------------------------------------------------------------
# §2.1  ExpertStatus — immutable snapshot for one expert in one layer
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertStatus:
    """Immutable diagnostic snapshot for a single expert within a router layer.

    Produced by :class:`CollapseDetector` and stored inside
    :class:`LayerCollapseReport`.

    Attributes
    ----------
    layer_name : str
        Fully-qualified module name of the parent router layer.
    expert_idx : int
        Zero-based index of the expert within the layer.
    state : ExpertState
        Current health classification (HEALTHY / COLD / DEAD / UNKNOWN).
    alert_level : AlertLevel
        Derived alert severity: DEAD→ERROR, COLD→WARN, HEALTHY→INFO,
        UNKNOWN→INFO (no data, not an error).
    utilization : float
        Fraction of total tokens routed to this expert in the current window.
        Range [0.0, 1.0].  ``float('nan')`` when the layer has no events.
    token_count : int
        Raw token count routed to this expert in the current window.
    consecutive_cold_steps : int
        Number of consecutive analysis calls during which the expert has been
        in the COLD or DEAD state. Used to distinguish temporary dips from
        confirmed long-term collapse. ``0`` for HEALTHY or UNKNOWN experts.
    """

    layer_name: str
    expert_idx: int
    state: ExpertState
    alert_level: AlertLevel
    utilization: float
    token_count: int
    consecutive_cold_steps: int

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """True when the expert is actively routing tokens."""
        return self.state == ExpertState.HEALTHY

    @property
    def is_cold(self) -> bool:
        """True when the expert is in the early-warning COLD state."""
        return self.state == ExpertState.COLD

    @property
    def is_dead(self) -> bool:
        """True when the expert is confirmed dead."""
        return self.state == ExpertState.DEAD

    @property
    def is_problematic(self) -> bool:
        """True when the expert is COLD or DEAD (any non-healthy state)."""
        return self.state in (ExpertState.COLD, ExpertState.DEAD)

    def to_dict(self) -> dict:
        """JSON-serialisable dictionary representation."""
        util = self.utilization
        return {
            "layer_name": self.layer_name,
            "expert_idx": self.expert_idx,
            "state": self.state.value,
            "alert_level": self.alert_level.value,
            "utilization": None if (util != util) else round(util, 8),  # NaN check
            "utilization_pct": None if (util != util) else round(util * 100, 4),
            "token_count": self.token_count,
            "consecutive_cold_steps": self.consecutive_cold_steps,
        }

    def __str__(self) -> str:
        util_str = (
            f"{self.utilization * 100:.3f}%"
            if self.utilization == self.utilization  # NaN check
            else "N/A"
        )
        cold_str = (
            f"  (cold {self.consecutive_cold_steps} steps)"
            if self.consecutive_cold_steps > 0
            else ""
        )
        return (
            f"Expert {self.expert_idx:>3d}  "
            f"util={util_str:<8s}  "
            f"[{self.state.value}]{cold_str}"
        )


# ------------------------------------------------------------------------------
# §2.2  LayerCollapseReport — collapse report for one layer
# ------------------------------------------------------------------------------


@dataclass
class LayerCollapseReport:
    """Expert collapse diagnostics for a single MoE router layer.

    Returned as values in the dict produced by
    :meth:`CollapseDetector.detect`.

    Attributes
    ----------
    layer_name : str
        Fully-qualified module name.
    n_experts : int
        Total number of experts in this layer.
    experts : list of ExpertStatus
        Per-expert status, ordered by expert index.
    n_dead : int
        Number of confirmed dead experts.
    n_cold : int
        Number of cold (early-warning) experts.
    n_healthy : int
        Number of healthy experts.
    global_alert : AlertLevel
        Worst alert level across all experts (INFO < WARN < ERROR).
    load_imbalance_score : float
        max_utilization / mean_utilization. 1.0 = perfect balance.
        ``float('nan')`` when the layer has no events.
    is_empty : bool
        True when no routing events have been collected yet.
    """

    layer_name: str
    n_experts: int
    experts: List[ExpertStatus] = field(default_factory=list)
    n_dead: int = 0
    n_cold: int = 0
    n_healthy: int = 0
    global_alert: AlertLevel = AlertLevel.INFO
    load_imbalance_score: float = float("nan")
    is_empty: bool = True

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_problematic(self) -> int:
        """Total number of non-healthy experts (n_dead + n_cold)."""
        return self.n_dead + self.n_cold

    @property
    def is_collapsed(self) -> bool:
        """True when at least one expert is confirmed DEAD."""
        return self.n_dead > 0

    @property
    def is_degrading(self) -> bool:
        """True when at least one expert is COLD or DEAD."""
        return self.n_problematic > 0

    def dead_experts(self) -> List[ExpertStatus]:
        """Return only the DEAD experts."""
        return [e for e in self.experts if e.is_dead]

    def cold_experts(self) -> List[ExpertStatus]:
        """Return only the COLD experts."""
        return [e for e in self.experts if e.is_cold]

    def healthy_experts(self) -> List[ExpertStatus]:
        """Return only the HEALTHY experts."""
        return [e for e in self.experts if e.is_healthy]

    def expert(self, idx: int) -> Optional[ExpertStatus]:
        """Return the ExpertStatus for expert index *idx*, or None."""
        if 0 <= idx < len(self.experts):
            return self.experts[idx]
        return None

    def to_dict(self) -> dict:
        """JSON-serialisable representation of this layer's collapse report."""
        lim = self.load_imbalance_score
        return {
            "layer_name": self.layer_name,
            "n_experts": self.n_experts,
            "n_dead": self.n_dead,
            "n_cold": self.n_cold,
            "n_healthy": self.n_healthy,
            "global_alert": self.global_alert.value,
            "load_imbalance_score": None if (lim != lim) else round(lim, 4),
            "is_empty": self.is_empty,
            "experts": [e.to_dict() for e in self.experts],
        }

    def __repr__(self) -> str:
        return (
            f"LayerCollapseReport("
            f"layer={self.layer_name!r}, "
            f"n_experts={self.n_experts}, "
            f"dead={self.n_dead}, "
            f"cold={self.n_cold}, "
            f"alert={self.global_alert.value}"
            f")"
        )


# =============================================================================
# Section 3 — CollapseDetector
# =============================================================================


class CollapseDetector:
    """Stateful expert collapse detector for MoE router diagnostics.

    Consumes per-layer routing statistics and classifies each expert as
    HEALTHY, COLD, or DEAD based on utilisation thresholds and persistence
    tracking (consecutive cold steps).

    Statefulness
    ------------
    ``CollapseDetector`` maintains a ``_cold_counters`` dict of the form:
    ``{layer_name: {expert_idx: consecutive_cold_steps}}``.

    On each call to :meth:`detect`:
      - An expert's counter is incremented if it is below ``cold_threshold``.
      - The counter is reset to 0 when the expert rises above ``cold_threshold``.
      - An expert that has been cold for more than ``cold_steps_limit`` steps
        is promoted from COLD to DEAD regardless of current utilisation.

    This persistence tracking prevents false-positive DEAD alerts during
    transient routing dips that naturally occur in early training.

    Thread safety
    -------------
    Not thread-safe. Call ``detect()`` from a single thread (the reporter
    thread). The ``StatCollector`` read is already protected by its own lock.

    Parameters
    ----------
    config : WatchConfig
        Provides ``dead_threshold``, ``cold_threshold``, and
        ``cold_steps_limit``.

    Examples
    --------
    >>> detector = CollapseDetector(config)
    >>> stats = collector.get_all_stats()
    >>> layer_reports = detector.detect(stats)
    >>> for layer, report in layer_reports.items():
    ...     if report.is_collapsed:
    ...         print(f"{layer}: {report.n_dead} dead expert(s)")
    """

    def __init__(self, config: WatchConfig) -> None:
        self.config = config

        # {layer_name: {expert_idx: consecutive_cold_steps}}
        self._cold_counters: Dict[str, Dict[int, int]] = {}

        log.debug(
            "[moewatch] CollapseDetector initialised "
            "(dead_threshold=%.4f, cold_threshold=%.4f, cold_steps_limit=%d).",
            config.dead_threshold,
            config.cold_threshold,
            config.cold_steps_limit,
        )

    # ==========================================================================
    # §3.1  Public interface
    # ==========================================================================

    def detect(
        self,
        all_stats: Dict[str, LayerStats],
    ) -> Dict[str, LayerCollapseReport]:
        """Run collapse detection over all tracked layers.

        Iterates over each layer's :class:`~moewatch.collector.stat_collector.LayerStats`,
        classifies every expert, updates cold-step counters, and returns one
        :class:`LayerCollapseReport` per layer.

        Parameters
        ----------
        all_stats : dict
            ``{layer_name: LayerStats}`` as returned by
            :meth:`~moewatch.collector.stat_collector.StatCollector.get_all_stats`.
            An empty dict is valid (returns an empty dict).

        Returns
        -------
        dict
            ``{layer_name: LayerCollapseReport}`` — same key set as *all_stats*.
        """
        if not all_stats:
            log.debug("[moewatch] CollapseDetector.detect() called with empty stats.")
            return {}

        reports: Dict[str, LayerCollapseReport] = {}
        total_dead = 0
        total_cold = 0

        for layer_name, stats in all_stats.items():
            report = self._detect_layer(layer_name, stats)
            reports[layer_name] = report
            total_dead += report.n_dead
            total_cold += report.n_cold

        if total_dead > 0:
            log.error(
                "[moewatch] CollapseDetector: %d dead expert(s) detected "
                "across %d layer(s). Intervention recommended.",
                total_dead,
                sum(1 for r in reports.values() if r.is_collapsed),
            )
        elif total_cold > 0:
            log.warning(
                "[moewatch] CollapseDetector: %d cold expert(s) detected "
                "across %d layer(s). Monitor closely.",
                total_cold,
                sum(1 for r in reports.values() if r.is_degrading),
            )
        else:
            log.debug(
                "[moewatch] CollapseDetector: all experts healthy across "
                "%d layer(s).",
                len(reports),
            )

        return reports

    def reset(self, layer_name: Optional[str] = None) -> None:
        """Reset cold-step counters for a specific layer or all layers.

        Call this when a model checkpoint is restored mid-training or when
        you want to restart collapse tracking from a clean state.

        Parameters
        ----------
        layer_name : str, optional
            If provided, reset only that layer. If ``None``, reset all layers.
        """
        if layer_name is not None:
            self._cold_counters.pop(layer_name, None)
            log.debug(
                "[moewatch] CollapseDetector: counters reset for layer %s.",
                layer_name,
            )
        else:
            self._cold_counters.clear()
            log.debug("[moewatch] CollapseDetector: all counters reset.")

    @property
    def tracked_layers(self) -> List[str]:
        """Names of layers that have at least one cold-step counter entry."""
        return list(self._cold_counters.keys())

    def cold_counter_for(self, layer_name: str, expert_idx: int) -> int:
        """Return the consecutive cold-step count for a specific expert.

        Returns 0 if the layer or expert has not been tracked.

        Parameters
        ----------
        layer_name : str
        expert_idx : int
        """
        return self._cold_counters.get(layer_name, {}).get(expert_idx, 0)

    # ==========================================================================
    # §3.2  Per-layer detection
    # ==========================================================================

    def _detect_layer(
        self,
        layer_name: str,
        stats: LayerStats,
    ) -> LayerCollapseReport:
        """Run collapse detection for a single layer.

        Returns
        -------
        LayerCollapseReport
        """
        # -- Empty layer or insufficient data -----------------------------------
        if (
            stats.is_empty
            or stats.n_experts == 0
            or stats.event_count < self.config.min_events_for_collapse
        ):
            return LayerCollapseReport(
                layer_name=layer_name,
                n_experts=stats.n_experts,
                experts=[],
                is_empty=True,
                global_alert=AlertLevel.INFO,
                load_imbalance_score=float("nan"),
            )

        n_experts = stats.n_experts
        utilization = stats.utilization  # (n_experts,) float32 CPU
        token_counts = stats.expert_counts  # (n_experts,) int64 CPU

        # -- Ensure cold counter dict exists for this layer ------------------
        if layer_name not in self._cold_counters:
            self._cold_counters[layer_name] = {}

        counters = self._cold_counters[layer_name]

        # -- Validate tensor dimensions --------------------------------------
        if utilization.shape[0] != n_experts:
            log.warning(
                "[moewatch] CollapseDetector(%s): utilization shape %s "
                "does not match n_experts=%d. Skipping.",
                layer_name,
                list(utilization.shape),
                n_experts,
            )
            return LayerCollapseReport(
                layer_name=layer_name,
                n_experts=n_experts,
                experts=[],
                is_empty=True,
                global_alert=AlertLevel.INFO,
                load_imbalance_score=float("nan"),
            )

        # -- Classify each expert --------------------------------------------
        expert_statuses: List[ExpertStatus] = []
        n_dead = 0
        n_cold = 0
        n_healthy = 0

        util_list = utilization.tolist()
        counts_list = token_counts.tolist()

        for idx in range(n_experts):
            util = util_list[idx]
            token_count = int(counts_list[idx])

            state, alert, new_counter = self._classify_expert(
                layer_name=layer_name,
                expert_idx=idx,
                utilization=util,
                current_cold_steps=counters.get(idx, 0),
            )

            # Update cold counter
            counters[idx] = new_counter

            status = ExpertStatus(
                layer_name=layer_name,
                expert_idx=idx,
                state=state,
                alert_level=alert,
                utilization=util,
                token_count=token_count,
                consecutive_cold_steps=new_counter,
            )
            expert_statuses.append(status)

            if state == ExpertState.DEAD:
                n_dead += 1
            elif state == ExpertState.COLD:
                n_cold += 1
            else:
                n_healthy += 1

        # -- Clean up counters for experts that no longer exist --------------
        # (handles architecture changes that reduce n_experts mid-run)
        stale_keys = [k for k in counters if k >= n_experts]
        for k in stale_keys:
            del counters[k]

        # -- Determine global alert level ------------------------------------
        global_alert = self._aggregate_alert(n_dead, n_cold)

        # -- Emit per-layer log ---------------------------------------------
        self._log_layer_report(
            layer_name, n_experts, n_dead, n_cold, stats.load_imbalance_score
        )

        return LayerCollapseReport(
            layer_name=layer_name,
            n_experts=n_experts,
            experts=expert_statuses,
            n_dead=n_dead,
            n_cold=n_cold,
            n_healthy=n_healthy,
            global_alert=global_alert,
            load_imbalance_score=stats.load_imbalance_score,
            is_empty=False,
        )

    # ==========================================================================
    # §3.3  Expert classification
    # ==========================================================================

    def _classify_expert(
        self,
        layer_name: str,
        expert_idx: int,
        utilization: float,
        current_cold_steps: int,
    ) -> Tuple[ExpertState, AlertLevel, int]:
        """Classify a single expert and update its cold-step counter.

        Classification rules
        --------------------
        1. utilisation ≤ dead_threshold:
             → DEAD immediately. Counter incremented.
        2. utilisation ≤ cold_threshold (but > dead_threshold):
             → cold counter incremented.
             → If counter now ≥ cold_steps_limit: DEAD (promoted from COLD).
             → Else: COLD.
        3. utilisation > cold_threshold:
             → HEALTHY. Counter reset to 0.

        Parameters
        ----------
        utilization : float
            Fraction of tokens routed to this expert in the current window.
        current_cold_steps : int
            Counter value from the previous call (0 if first time seen).

        Returns
        -------
        (ExpertState, AlertLevel, new_cold_counter)
        """
        dead_thresh = self.config.dead_threshold
        cold_thresh = self.config.cold_threshold
        cold_limit = self.config.cold_steps_limit

        # -- Rule 1: confirmed dead by utilisation threshold ------------------
        if utilization <= dead_thresh:
            new_counter = current_cold_steps + 1
            log.debug(
                "[moewatch] CollapseDetector(%s): Expert %d DEAD "
                "(util=%.4f%% ≤ dead_threshold=%.4f%%).",
                layer_name,
                expert_idx,
                utilization * 100,
                dead_thresh * 100,
            )
            return ExpertState.DEAD, AlertLevel.ERROR, new_counter

        # -- Rule 2a: below cold threshold — increment counter ----------------
        if utilization <= cold_thresh:
            new_counter = current_cold_steps + 1

            # -- Rule 2b: cold too long → promote to DEAD --------------------
            if new_counter >= cold_limit:
                log.debug(
                    "[moewatch] CollapseDetector(%s): Expert %d promoted COLD→DEAD "
                    "(cold for %d steps ≥ cold_steps_limit=%d).",
                    layer_name,
                    expert_idx,
                    new_counter,
                    cold_limit,
                )
                return ExpertState.DEAD, AlertLevel.ERROR, new_counter

            log.debug(
                "[moewatch] CollapseDetector(%s): Expert %d COLD "
                "(util=%.4f%%, cold_steps=%d/%d).",
                layer_name,
                expert_idx,
                utilization * 100,
                new_counter,
                cold_limit,
            )
            return ExpertState.COLD, AlertLevel.WARN, new_counter

        # -- Rule 3: healthy — reset cold counter ----------------------------
        if current_cold_steps > 0:
            log.debug(
                "[moewatch] CollapseDetector(%s): Expert %d recovered to HEALTHY "
                "(was cold for %d steps).",
                layer_name,
                expert_idx,
                current_cold_steps,
            )
        return ExpertState.HEALTHY, AlertLevel.INFO, 0

    # ==========================================================================
    # §3.4  Aggregate alert helper
    # ==========================================================================

    @staticmethod
    def _aggregate_alert(n_dead: int, n_cold: int) -> AlertLevel:
        """Derive global alert level from expert counts.

        ERROR if any expert is dead; WARN if any are cold; INFO otherwise.
        """
        if n_dead > 0:
            return AlertLevel.ERROR
        if n_cold > 0:
            return AlertLevel.WARN
        return AlertLevel.INFO

    # ==========================================================================
    # §3.5  Load imbalance analysis
    # ==========================================================================

    @staticmethod
    def compute_load_imbalance(utilization: torch.Tensor) -> Tuple[float, int, float]:
        """Compute the load imbalance score for a given utilisation tensor.

        The load imbalance score is defined as max_util / mean_util.
        A perfectly balanced layer scores 1.0; a fully collapsed layer
        (one expert receives all tokens) scores n_experts.

        Parameters
        ----------
        utilization : torch.Tensor
            1-D float tensor of shape ``(n_experts,)`` — expert token fractions.
            Must sum to ≤ 1.0 (may be less than 1.0 for top-k routing where
            the sum equals top_k).

        Returns
        -------
        (load_imbalance_score, argmax_expert, max_utilization)
            ``load_imbalance_score``: float — max/mean ratio.
            ``argmax_expert``:        int   — index of the most-loaded expert.
            ``max_utilization``:      float — utilisation fraction of argmax_expert.

        Notes
        -----
        Returns ``(float('nan'), -1, float('nan'))`` for empty or all-zero
        tensors.
        """
        if utilization.numel() == 0:
            return float("nan"), -1, float("nan")

        util = utilization.float().cpu()
        total = util.sum().item()

        if total <= 0.0:
            return float("nan"), -1, float("nan")

        mean_util = util.mean().item()
        max_val, argmax = util.max(dim=0)
        max_util = max_val.item()
        argmax_idx = int(argmax.item())

        if mean_util <= 0.0:
            return float("nan"), argmax_idx, max_util

        score = max_util / mean_util
        return float(score), argmax_idx, float(max_util)

    # ==========================================================================
    # §3.6  Logging helpers
    # ==========================================================================

    def _log_layer_report(
        self,
        layer_name: str,
        n_experts: int,
        n_dead: int,
        n_cold: int,
        load_imbalance_score: float,
    ) -> None:
        """Emit a single structured log line per layer."""
        lim_str = (
            f"{load_imbalance_score:.2f}x"
            if load_imbalance_score == load_imbalance_score  # NaN check
            else "N/A"
        )

        if n_dead > 0:
            log.error(
                "[moewatch] COLLAPSE DETECTED  %-50s  "
                "dead=%d/%d  cold=%d  imbalance=%s",
                layer_name,
                n_dead,
                n_experts,
                n_cold,
                lim_str,
            )
        elif n_cold > 0:
            log.warning(
                "[moewatch] COLLAPSE WARNING   %-50s  "
                "dead=%d/%d  cold=%d  imbalance=%s",
                layer_name,
                n_dead,
                n_experts,
                n_cold,
                lim_str,
            )
        else:
            log.debug(
                "[moewatch] experts healthy    %-50s  "
                "dead=%d/%d  cold=%d  imbalance=%s",
                layer_name,
                n_dead,
                n_experts,
                n_cold,
                lim_str,
            )

    # ==========================================================================
    # §3.7  Dunder methods
    # ==========================================================================

    def __repr__(self) -> str:
        tracked = len(self._cold_counters)
        total_tracked = sum(len(v) for v in self._cold_counters.values())
        return (
            f"CollapseDetector("
            f"dead_threshold={self.config.dead_threshold}, "
            f"cold_threshold={self.config.cold_threshold}, "
            f"cold_steps_limit={self.config.cold_steps_limit}, "
            f"tracked_layers={tracked}, "
            f"tracked_experts={total_tracked}"
            f")"
        )
