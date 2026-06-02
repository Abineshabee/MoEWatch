# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  report/audit_report.py — Structured result object returned by audit()
#
#  AuditReport is a pure data container. It holds the complete diagnostic
#  output of a single audit() run — entropy results, collapse reports, raw
#  layer statistics, config snapshot, and run metadata — and exposes clean
#  query methods that the researcher can call after the fact.
#
#  Design principles
#  -----------------
#    - Immutable after construction: all public attributes are set once in
#      __init__. No mutable state. Callers can safely share and cache reports.
#    - Zero side effects: none of the query methods print, log, or modify state.
#      Formatting and printing live entirely in CLIReporter.
#    - Always returns something: dead_experts(), routing_entropy(), and
#      utilization() return empty structures (list / dict) — never None —
#      so callers can iterate without guarding.
#    - JSON-round-trippable: to_json() / from_json() preserve all numeric
#      data faithfully; tensor data is serialised as Python lists.
#
#  Consumers
#  ---------
#    - audit()          → constructs AuditReport and optionally passes it to
#                         CLIReporter.print_summary().
#    - Users            → call report.summary(), report.dead_experts(), etc.
#    - CI pipelines     → call report.to_json() and parse the output.
#    - MoEWatch.step()  → does NOT produce AuditReport (live path uses Alerts).
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#  Version: 0.1.0
#
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..config import AlertLevel, WatchConfig
from ..collector.stat_collector import LayerStats
from ..analyzer.entropy import LayerEntropyReport, EntropyResult
from ..analyzer.collapse import LayerCollapseReport, ExpertState

log = logging.getLogger(__name__)


# =============================================================================
# Section 1 — Overall health classification
# =============================================================================

class OverallHealth(str, Enum):
    """Top-level routing health classification for the entire model.

    Derived from the union of all per-layer entropy and collapse reports.
    Used as the first signal in AuditReport.summary() and to_json().

    Attributes
    ----------
    HEALTHY:
        All experts active, entropy above warn threshold on every layer.
        No action required.
    DEGRADING:
        At least one expert is COLD or one layer's entropy is in WARN range.
        Monitor closely; consider adjusting aux_loss_coef if trend continues.
    CRITICAL:
        At least one expert is DEAD or one layer's entropy is in ERROR range.
        Immediate intervention recommended (aux loss, learning rate, restart).
    UNKNOWN:
        No events were collected (model produced no forward passes during audit).
        Check that the dataloader and model are correctly configured.
    """

    HEALTHY   = "HEALTHY"
    DEGRADING = "DEGRADING"
    CRITICAL  = "CRITICAL"
    UNKNOWN   = "UNKNOWN"


# =============================================================================
# Section 2 — DeadExpertEntry (flat, user-facing summary record)
# =============================================================================

@dataclass(frozen=True)
class DeadExpertEntry:
    """Flat, user-facing record for a single dead or cold expert.

    Produced by :meth:`AuditReport.dead_experts` — a flat list is much easier
    to iterate in user scripts than the nested LayerCollapseReport hierarchy.

    Attributes
    ----------
    layer_name : str
        Fully-qualified module name of the router layer.
    expert_idx : int
        Zero-based expert index within the layer.
    state : ExpertState
        DEAD or COLD.
    utilization : float
        Fraction of tokens routed to this expert in the audit window.
    utilization_pct : float
        ``utilization * 100`` — convenience field for display.
    consecutive_cold_steps : int
        Number of analysis calls during which the expert has been non-healthy.
    layer_idx : int
        Positional index of this layer in the ordered router list (0-based).
        Useful for correlating with model architecture depth.
    """

    layer_name:             str
    expert_idx:             int
    state:                  ExpertState
    utilization:            float
    utilization_pct:        float
    consecutive_cold_steps: int
    layer_idx:              int

    @property
    def is_dead(self) -> bool:
        return self.state == ExpertState.DEAD

    @property
    def is_cold(self) -> bool:
        return self.state == ExpertState.COLD

    def to_dict(self) -> dict:
        return {
            "layer_name":             self.layer_name,
            "layer_idx":              self.layer_idx,
            "expert_idx":             self.expert_idx,
            "state":                  self.state.value,
            "utilization":            round(self.utilization, 8),
            "utilization_pct":        round(self.utilization_pct, 4),
            "consecutive_cold_steps": self.consecutive_cold_steps,
        }

    def __str__(self) -> str:
        icon = "❌" if self.is_dead else "⚠ "
        return (
            f"{icon} Layer {self.layer_idx:>2d}  Expert {self.expert_idx:>3d}  "
            f"util: {self.utilization_pct:.3f}%  [{self.state.value}]"
        )


# =============================================================================
# Section 3 — UtilizationSummary (per-layer token-distribution snapshot)
# =============================================================================

@dataclass(frozen=True)
class UtilizationSummary:
    """Expert token-distribution summary for a single router layer.

    Produced by :meth:`AuditReport.utilization`. Groups raw utilization
    tensor data with derived statistics for easy display and export.

    Attributes
    ----------
    layer_name : str
        Fully-qualified module name.
    layer_idx : int
        Positional index in the ordered router list.
    n_experts : int
        Total number of experts.
    utilization : list of float
        Per-expert token fraction (0.0 – 1.0), length == n_experts.
    utilization_pct : list of float
        Per-expert token percentage (0.0 – 100.0).
    token_counts : list of int
        Raw per-expert token counts over the audit window.
    total_tokens : int
        Sum of all token counts.
    max_util : float
        Utilization of the most-loaded expert.
    min_util : float
        Utilization of the least-loaded expert (may be 0.0 for dead experts).
    mean_util : float
        Mean expert utilization (== 1 / n_experts for perfect balance).
    load_imbalance_score : float
        max_util / mean_util. 1.0 = perfect balance.
    event_count : int
        Number of RoutingEvent objects that contributed to these stats.
    step_range : tuple of (int, int)
        (first_step, last_step) of the contributing events.
    """

    layer_name:           str
    layer_idx:            int
    n_experts:            int
    utilization:          List[float]
    utilization_pct:      List[float]
    token_counts:         List[int]
    total_tokens:         int
    max_util:             float
    min_util:             float
    mean_util:            float
    load_imbalance_score: float
    event_count:          int
    step_range:           Tuple[int, int]

    def to_dict(self) -> dict:
        lim = self.load_imbalance_score
        return {
            "layer_name":           self.layer_name,
            "layer_idx":            self.layer_idx,
            "n_experts":            self.n_experts,
            "utilization":          [round(u, 8) for u in self.utilization],
            "utilization_pct":      [round(u, 4) for u in self.utilization_pct],
            "token_counts":         self.token_counts,
            "total_tokens":         self.total_tokens,
            "max_util":             round(self.max_util, 8),
            "min_util":             round(self.min_util, 8),
            "mean_util":            round(self.mean_util, 8),
            "load_imbalance_score": None if math.isnan(lim) else round(lim, 4),
            "event_count":          self.event_count,
            "step_range":           list(self.step_range),
        }


# =============================================================================
# Section 4 — AuditReport
# =============================================================================

class AuditReport:
    """Structured diagnostic result object returned by :func:`~moewatch.audit`.

    Holds the complete output of a single audit run: per-layer entropy results,
    per-layer collapse reports, raw layer statistics, configuration snapshot,
    and run metadata. Exposes clean query methods for common access patterns.

    This object is **immutable after construction** — all fields are set in
    ``__init__`` and none of the query methods have side effects. It is safe
    to cache, pass between threads, or serialise to JSON.

    Parameters
    ----------
    layer_stats : dict
        ``{layer_name: LayerStats}`` from StatCollector.get_all_stats().
    entropy_results : LayerEntropyReport
        Per-layer entropy analysis from EntropyAnalyzer.analyze().
    collapse_results : dict
        ``{layer_name: LayerCollapseReport}`` from CollapseDetector.detect().
    config : WatchConfig
        Configuration used for this audit run (snapshot, not a live reference).
    completed_steps : int
        Number of forward passes that actually ran (may be < requested steps
        if the dataloader was exhausted early).
    elapsed_seconds : float
        Wall-clock time the audit loop took (excludes report construction).
    device : str
        String representation of the device used (e.g. ``"cuda:0"``, ``"cpu"``).
    router_module_names : list of str
        Ordered list of all router module names that were instrumented.

    Examples
    --------
    Minimal usage:

    >>> report = audit(model)
    >>> report.summary()                # prints full human-readable report
    >>> dead = report.dead_experts()    # list[DeadExpertEntry]
    >>> entropy = report.routing_entropy()  # dict[layer_name, EntropyResult]
    >>> util = report.utilization()     # dict[layer_name, UtilizationSummary]
    >>> data = report.to_json()         # str — machine-readable JSON

    Programmatic use:

    >>> if report.overall_health == OverallHealth.CRITICAL:
    ...     raise RuntimeError("Expert collapse detected — stopping training.")
    """

    # =========================================================================
    # §4.1  Construction
    # =========================================================================

    def __init__(
        self,
        *,
        layer_stats:         Dict[str, LayerStats],
        entropy_results:     LayerEntropyReport,
        collapse_results:    Dict[str, LayerCollapseReport],
        config:              WatchConfig,
        completed_steps:     int,
        elapsed_seconds:     float,
        device:              str,
        router_module_names: List[str],
    ) -> None:

        # ── Raw results from the analyzer layer ──────────────────────────────
        self._layer_stats:      Dict[str, LayerStats]         = layer_stats
        self._entropy_results:  LayerEntropyReport            = entropy_results
        self._collapse_results: Dict[str, LayerCollapseReport] = collapse_results

        # ── Configuration snapshot ────────────────────────────────────────────
        self.config: WatchConfig = config

        # ── Run metadata ──────────────────────────────────────────────────────
        self.completed_steps:     int   = completed_steps
        self.elapsed_seconds:     float = elapsed_seconds
        self.device:              str   = device
        self.router_module_names: List[str] = list(router_module_names)
        self.created_at:          float = time.time()   # wall-clock creation timestamp

        # ── Derived top-level health (computed once, cached) ─────────────────
        self._overall_health: OverallHealth = self._compute_overall_health()

        # ── Pre-computed flat collections (cached on first access) ────────────
        self.__dead_experts_cache:  Optional[List[DeadExpertEntry]]         = None
        self.__utilization_cache:   Optional[Dict[str, UtilizationSummary]] = None

        log.debug(
            "[moewatch] AuditReport constructed: %d layers, %d steps, health=%s.",
            len(router_module_names),
            completed_steps,
            self._overall_health.value,
        )

    # =========================================================================
    # §4.2  Top-level health property
    # =========================================================================

    @property
    def overall_health(self) -> OverallHealth:
        """Top-level routing health classification for the entire model.

        Derived from the union of all per-layer entropy and collapse results.
        See :class:`OverallHealth` for the four possible values.
        """
        return self._overall_health

    def _compute_overall_health(self) -> OverallHealth:
        """Derive OverallHealth from entropy and collapse results.

        Priority (worst wins):
          CRITICAL > DEGRADING > HEALTHY > UNKNOWN
        """
        has_any_data = any(
            not stats.is_empty for stats in self._layer_stats.values()
        )
        if not has_any_data:
            return OverallHealth.UNKNOWN

        # Check collapse results (dead experts → CRITICAL, cold → DEGRADING)
        for report in self._collapse_results.values():
            if report.is_collapsed:
                return OverallHealth.CRITICAL

        for report in self._collapse_results.values():
            if report.is_degrading:
                # May still be upgraded to CRITICAL by entropy — keep scanning
                if self._entropy_results.global_alert == AlertLevel.ERROR:
                    return OverallHealth.CRITICAL
                return OverallHealth.DEGRADING

        # Check entropy results
        if self._entropy_results.global_alert == AlertLevel.ERROR:
            return OverallHealth.CRITICAL

        if self._entropy_results.global_alert == AlertLevel.WARN:
            return OverallHealth.DEGRADING

        return OverallHealth.HEALTHY

    # =========================================================================
    # §4.3  Primary query methods
    # =========================================================================

    def dead_experts(
        self,
        *,
        include_cold: bool = True,
    ) -> List[DeadExpertEntry]:
        """Return a flat list of all dead (and optionally cold) experts.

        Results are ordered by (layer_idx, expert_idx).

        Parameters
        ----------
        include_cold : bool
            When True (default), includes COLD experts as well as DEAD ones.
            When False, returns only confirmed DEAD experts.

        Returns
        -------
        list of DeadExpertEntry
            Empty list when all experts are healthy. Never returns None.

        Examples
        --------
        >>> dead = report.dead_experts()
        >>> for entry in dead:
        ...     print(entry)
        ❌ Layer  0  Expert   7  util: 0.010%  [DEAD]
        ⚠  Layer  0  Expert  12  util: 0.300%  [COLD]

        >>> # Only confirmed-dead, not cold:
        >>> confirmed = report.dead_experts(include_cold=False)
        """
        if self.__dead_experts_cache is not None:
            entries = self.__dead_experts_cache
        else:
            entries = self._build_dead_experts_list()
            self.__dead_experts_cache = entries

        if include_cold:
            return list(entries)

        return [e for e in entries if e.is_dead]

    def routing_entropy(self) -> Dict[str, EntropyResult]:
        """Return per-layer entropy results.

        Returns
        -------
        dict
            ``{layer_name: EntropyResult}`` — one entry per instrumented router
            layer.  Empty dict when no events were collected.  Never returns None.

        Examples
        --------
        >>> entropy = report.routing_entropy()
        >>> for name, result in entropy.items():
        ...     print(f"{name}: {result.entropy_norm:.1%} of max [{result.alert_level.value}]")
        model.layers.0.block_sparse_moe: 92.4% of max [INFO]
        model.layers.1.block_sparse_moe: 41.0% of max [ERROR]
        """
        return dict(self._entropy_results.results)

    def utilization(self) -> Dict[str, UtilizationSummary]:
        """Return per-expert token distribution for every layer.

        Returns
        -------
        dict
            ``{layer_name: UtilizationSummary}`` — one entry per instrumented
            router layer.  Empty dict when no events were collected.  Never
            returns None.

        Examples
        --------
        >>> util = report.utilization()
        >>> for name, u in util.items():
        ...     print(f"{name}: imbalance={u.load_imbalance_score:.2f}x")
        """
        if self.__utilization_cache is not None:
            return dict(self.__utilization_cache)

        cache = self._build_utilization_dict()
        self.__utilization_cache = cache
        return dict(cache)

    def recommendations(self) -> List[str]:
        """Return a prioritised list of actionable fix suggestions.

        Suggestions are derived from the current diagnostic state and ordered
        from most severe to least severe. This method is a v0.1 preview —
        suggestions are heuristic. A full recommendations engine is planned
        for v0.2.

        Returns
        -------
        list of str
            Each string is a self-contained, actionable recommendation.
            Empty list when the model is healthy.

        Examples
        --------
        >>> for rec in report.recommendations():
        ...     print(f"  💡 {rec}")
        """
        recs: List[str] = []

        dead_list  = self.dead_experts(include_cold=False)
        cold_list  = self.dead_experts(include_cold=True)
        cold_only  = [e for e in cold_list if e.is_cold]
        low_entropy_layers = [
            r for r in self._entropy_results.results.values()
            if r.alert_level in (AlertLevel.WARN, AlertLevel.ERROR)
            and not r.is_empty
        ]
        declining_layers = [
            r for r in self._entropy_results.results.values()
            if r.is_declining and not r.is_empty
        ]

        # -- Dead expert recommendations ------------------------------------
        if dead_list:
            n = len(dead_list)
            layer_names = sorted({e.layer_name for e in dead_list})
            recs.append(
                f"[CRITICAL] {n} dead expert(s) detected in "
                f"{len(layer_names)} layer(s) "
                f"({', '.join(layer_names[:2])}{'...' if len(layer_names) > 2 else ''}). "
                "Raise aux_loss_coef (e.g. 0.01 → 0.04) to increase load-balancing "
                "pressure. If using Mixtral, check router_aux_loss_coef in the model "
                "config. If experts remain dead after adjustment, consider restarting "
                "from an earlier checkpoint."
            )

        # -- Low entropy recommendations -----------------------------------
        critical_entropy = [r for r in low_entropy_layers if r.is_critical]
        warn_entropy     = [r for r in low_entropy_layers if not r.is_critical]

        if critical_entropy:
            worst  = min(critical_entropy, key=lambda r: r.entropy_norm)
            recs.append(
                f"[CRITICAL] Routing entropy in ERROR zone: "
                f"{worst.entropy_norm:.1%} of max on layer '{worst.layer_name}'. "
                "The router is severely collapsed. Possible causes: (1) aux loss "
                "coefficient too low, (2) router learning rate too high relative "
                "to expert LR, (3) gradient explosion at this step. Check your "
                "loss curve for spikes."
            )
        elif warn_entropy:
            recs.append(
                f"[WARN] Low routing entropy on {len(warn_entropy)} layer(s). "
                "Router is under-exploring the expert space. Consider raising "
                "aux_loss_coef by 2× and monitoring for 500 more steps."
            )

        # -- Declining trend recommendation --------------------------------
        if declining_layers and not critical_entropy:
            worst_decline = min(declining_layers, key=lambda r: r.trend_delta)
            recs.append(
                f"[WARN] Entropy DECLINING on {len(declining_layers)} layer(s) "
                f"(worst: '{worst_decline.layer_name}', "
                f"delta={worst_decline.trend_delta:.1%}). "
                "Collapse is not yet critical but is trending in the wrong direction. "
                "This is the ideal time to intervene — act before experts go cold."
            )

        # -- Cold expert recommendation ------------------------------------
        if cold_only and not dead_list:
            n = len(cold_only)
            recs.append(
                f"[WARN] {n} cold expert(s) detected. These experts may recover "
                "naturally in early training, or may be precursors to collapse. "
                "If they remain cold for more than "
                f"{self.config.cold_steps_limit} analysis steps, they will be "
                "flagged as dead. Monitor closely."
            )

        # -- Load imbalance recommendation ---------------------------------
        high_imbalance = [
            (name, stats.load_imbalance_score)
            for name, stats in self._layer_stats.items()
            if not math.isnan(stats.load_imbalance_score)
            and stats.load_imbalance_score > self.config.load_imbalance_error
        ]
        if high_imbalance:
            worst_name, worst_score = max(high_imbalance, key=lambda x: x[1])
            recs.append(
                f"[WARN] Expert load imbalance critical on '{worst_name}' "
                f"(max/mean = {worst_score:.1f}x). For token-choice routing "
                "(top-k), consider adding a capacity factor or enabling "
                "router_jitter_noise in the model config."
            )

        # -- Healthy model ------------------------------------------------
        if not recs:
            recs.append(
                "[INFO] All experts active, entropy healthy on all layers. "
                "No action required."
            )

        return recs

    def summary(self) -> str:
        """Return a concise multi-line human-readable summary string.

        Unlike ``CLIReporter.print_summary()`` (which renders a full coloured
        report to stdout), this method returns a plain-text string — useful
        for logging, notebook display, or CI output without ANSI codes.

        Returns
        -------
        str
            Multi-line summary string.  Never None.

        Examples
        --------
        >>> print(report.summary())
        ═══════════════════════════════════════════════════════════
         moewatch Audit Summary
        ═══════════════════════════════════════════════════════════
         Overall health  : CRITICAL
         Router layers   : 32
         Completed steps : 50
         Elapsed         : 12.4 s
         Device          : cuda:0
        ...
        """
        lines: List[str] = []
        sep = "═" * 63

        lines.append(sep)
        lines.append(" moewatch Audit Summary")
        lines.append(sep)
        lines.append(f" Overall health  : {self._overall_health.value}")
        lines.append(f" Router layers   : {len(self.router_module_names)}")
        lines.append(f" Completed steps : {self.completed_steps}")
        lines.append(f" Elapsed         : {self.elapsed_seconds:.1f} s")
        lines.append(f" Device          : {self.device}")
        lines.append("")

        # -- Entropy summary ------------------------------------------------
        lines.append(" Entropy")
        lines.append(" " + "─" * 61)
        entropy_results = self._entropy_results.results
        if not entropy_results:
            lines.append("   (no data collected)")
        else:
            for i, (name, result) in enumerate(entropy_results.items()):
                tag = "[ERROR]" if result.is_critical else ("[WARN]" if not result.is_healthy else "[OK]  ")
                short_name = _shorten_layer_name(name, max_len=42)
                if result.is_empty:
                    lines.append(f"   {tag}  {short_name:<42s}  NO DATA")
                else:
                    lines.append(
                        f"   {tag}  {short_name:<42s}  "
                        f"H={result.entropy_bits:.2f} bits  "
                        f"({result.entropy_norm*100:.1f}% of max)  "
                        f"{result.trend}"
                    )
        lines.append("")

        # -- Collapse summary -----------------------------------------------
        dead_all  = self.dead_experts(include_cold=True)
        dead_only = [e for e in dead_all if e.is_dead]
        cold_only = [e for e in dead_all if e.is_cold]

        lines.append(" Expert Collapse")
        lines.append(" " + "─" * 61)
        if not dead_all:
            lines.append("   [OK]  All experts healthy — no collapse detected.")
        else:
            if dead_only:
                lines.append(f"   [ERROR]  {len(dead_only)} dead expert(s):")
                for entry in dead_only[:10]:
                    short = _shorten_layer_name(entry.layer_name, max_len=40)
                    lines.append(
                        f"     ❌  {short}  Expert {entry.expert_idx:>3d}  "
                        f"util={entry.utilization_pct:.3f}%"
                    )
                if len(dead_only) > 10:
                    lines.append(f"     ... and {len(dead_only) - 10} more")
            if cold_only:
                lines.append(f"   [WARN]   {len(cold_only)} cold expert(s):")
                for entry in cold_only[:5]:
                    short = _shorten_layer_name(entry.layer_name, max_len=40)
                    lines.append(
                        f"     ⚠   {short}  Expert {entry.expert_idx:>3d}  "
                        f"util={entry.utilization_pct:.3f}%"
                    )
                if len(cold_only) > 5:
                    lines.append(f"     ... and {len(cold_only) - 5} more")
        lines.append("")

        # -- Recommendations ------------------------------------------------
        recs = self.recommendations()
        lines.append(" Recommendations")
        lines.append(" " + "─" * 61)
        for rec in recs:
            # Word-wrap to 60 chars
            wrapped = _word_wrap(rec, width=58, indent="   ")
            lines.append(wrapped)
        lines.append("")
        lines.append(sep)

        return "\n".join(lines)

    # =========================================================================
    # §4.4  Per-layer accessors
    # =========================================================================

    def layer_names(self) -> List[str]:
        """Return the ordered list of instrumented router module names."""
        return list(self.router_module_names)

    def layer_stats(self, layer_name: str) -> Optional[LayerStats]:
        """Return raw :class:`~moewatch.collector.stat_collector.LayerStats` for a layer.

        Parameters
        ----------
        layer_name : str
            Fully-qualified module name.

        Returns
        -------
        LayerStats or None
            None if the layer name is not in this report.
        """
        return self._layer_stats.get(layer_name)

    def entropy_for(self, layer_name: str) -> Optional[EntropyResult]:
        """Return the :class:`~moewatch.analyzer.entropy.EntropyResult` for a single layer.

        Parameters
        ----------
        layer_name : str
            Fully-qualified module name.

        Returns
        -------
        EntropyResult or None
        """
        return self._entropy_results.results.get(layer_name)

    def collapse_for(self, layer_name: str) -> Optional[LayerCollapseReport]:
        """Return the :class:`~moewatch.analyzer.collapse.LayerCollapseReport` for a layer.

        Parameters
        ----------
        layer_name : str
            Fully-qualified module name.

        Returns
        -------
        LayerCollapseReport or None
        """
        return self._collapse_results.get(layer_name)

    # =========================================================================
    # §4.5  Aggregate statistics
    # =========================================================================

    @property
    def n_layers(self) -> int:
        """Total number of instrumented router layers."""
        return len(self.router_module_names)

    @property
    def n_dead(self) -> int:
        """Total number of confirmed dead experts across all layers."""
        return sum(r.n_dead for r in self._collapse_results.values())

    @property
    def n_cold(self) -> int:
        """Total number of cold (early-warning) experts across all layers."""
        return sum(r.n_cold for r in self._collapse_results.values())

    @property
    def n_healthy(self) -> int:
        """Total number of healthy experts across all layers."""
        return sum(r.n_healthy for r in self._collapse_results.values())

    @property
    def n_experts_total(self) -> int:
        """Total expert count across all layers (n_dead + n_cold + n_healthy)."""
        return sum(r.n_experts for r in self._collapse_results.values())

    @property
    def worst_entropy_layer(self) -> Optional[EntropyResult]:
        """EntropyResult with the lowest normalised entropy, or None if no data."""
        return self._entropy_results.worst_layer()

    @property
    def has_collapse(self) -> bool:
        """True when at least one expert is confirmed DEAD."""
        return self.n_dead > 0

    @property
    def has_warnings(self) -> bool:
        """True when at least one expert is COLD or one layer's entropy is in WARN."""
        return (
            self.n_cold > 0
            or self._entropy_results.n_warn > 0
        )

    # =========================================================================
    # §4.6  Serialisation
    # =========================================================================

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full report to a JSON string.

        All tensor data is converted to Python lists. NaN values are
        converted to ``null``. The result is safe to write to disk or
        send over HTTP.

        Parameters
        ----------
        indent : int
            JSON indentation level. Default 2. Pass 0 for compact output.

        Returns
        -------
        str
            UTF-8 JSON string.  All values are JSON-serialisable.

        Examples
        --------
        >>> with open("moewatch_report.json", "w") as f:
        ...     f.write(report.to_json())
        """
        return json.dumps(self._to_dict(), indent=indent, ensure_ascii=False)

    def to_json_file(self, path: str, indent: int = 2) -> None:
        """Write the full report to a JSON file at *path*.

        Parameters
        ----------
        path : str
            File system path for the output file. Parent directory must exist.
        indent : int
            JSON indentation level. Default 2.

        Examples
        --------
        >>> report.to_json_file("./reports/audit_2026_06.json")
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json(indent=indent))
        log.info("[moewatch] AuditReport written to: %s", path)

    def _to_dict(self) -> dict:
        """Build the full JSON-serialisable dictionary representation."""

        # -- Run metadata ---------------------------------------------------
        meta: Dict[str, Any] = {
            "moewatch_version":    "0.1.0",
            "overall_health":      self._overall_health.value,
            "completed_steps":     self.completed_steps,
            "elapsed_seconds":     round(self.elapsed_seconds, 3),
            "device":              self.device,
            "created_at":          self.created_at,
            "router_module_names": self.router_module_names,
            "n_layers":            self.n_layers,
            "n_experts_total":     self.n_experts_total,
            "n_dead":              self.n_dead,
            "n_cold":              self.n_cold,
            "n_healthy":           self.n_healthy,
        }

        # -- Config snapshot ------------------------------------------------
        config_dict = self.config.to_dict()

        # -- Entropy results ------------------------------------------------
        entropy_dict = self._entropy_results.to_dict()

        # -- Collapse results -----------------------------------------------
        collapse_dict = {
            name: report.to_dict()
            for name, report in self._collapse_results.items()
        }

        # -- Flat dead experts list -----------------------------------------
        dead_experts_list = [
            entry.to_dict() for entry in self.dead_experts(include_cold=True)
        ]

        # -- Utilization summaries ------------------------------------------
        util_dict = {
            name: summary.to_dict()
            for name, summary in self.utilization().items()
        }

        # -- Recommendations ------------------------------------------------
        recs = self.recommendations()

        return {
            "metadata":       meta,
            "config":         config_dict,
            "entropy":        entropy_dict,
            "collapse":       collapse_dict,
            "dead_experts":   dead_experts_list,
            "utilization":    util_dict,
            "recommendations": recs,
        }

    # =========================================================================
    # §4.7  Internal builders
    # =========================================================================

    def _build_dead_experts_list(self) -> List[DeadExpertEntry]:
        """Build the flat dead-expert list from collapse results.

        Called once and cached. Results are sorted by (layer_idx, expert_idx)
        so the output is deterministic regardless of dict ordering.
        """
        entries: List[DeadExpertEntry] = []

        for layer_idx, layer_name in enumerate(self.router_module_names):
            report = self._collapse_results.get(layer_name)
            if report is None or report.is_empty:
                continue

            for expert_status in report.experts:
                if expert_status.is_problematic:
                    util = expert_status.utilization
                    entries.append(DeadExpertEntry(
                        layer_name=layer_name,
                        expert_idx=expert_status.expert_idx,
                        state=expert_status.state,
                        utilization=util,
                        utilization_pct=util * 100.0,
                        consecutive_cold_steps=expert_status.consecutive_cold_steps,
                        layer_idx=layer_idx,
                    ))

        entries.sort(key=lambda e: (e.layer_idx, e.expert_idx))
        return entries

    def _build_utilization_dict(self) -> Dict[str, UtilizationSummary]:
        """Build UtilizationSummary for every layer from raw LayerStats.

        Called once and cached.
        """
        result: Dict[str, UtilizationSummary] = {}

        for layer_idx, layer_name in enumerate(self.router_module_names):
            stats = self._layer_stats.get(layer_name)
            if stats is None or stats.is_empty or stats.n_experts == 0:
                continue

            util_list   = stats.utilization.tolist()
            counts_list = stats.expert_counts.tolist()
            total_tokens = stats.total_tokens

            max_u  = max(util_list) if util_list else 0.0
            min_u  = min(util_list) if util_list else 0.0
            mean_u = (sum(util_list) / len(util_list)) if util_list else 0.0
            lim    = stats.load_imbalance_score

            result[layer_name] = UtilizationSummary(
                layer_name=layer_name,
                layer_idx=layer_idx,
                n_experts=stats.n_experts,
                utilization=util_list,
                utilization_pct=[u * 100.0 for u in util_list],
                token_counts=[int(c) for c in counts_list],
                total_tokens=total_tokens,
                max_util=max_u,
                min_util=min_u,
                mean_util=mean_u,
                load_imbalance_score=lim,
                event_count=stats.event_count,
                step_range=stats.step_range,
            )

        return result

    # =========================================================================
    # §4.8  Dunder methods
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"AuditReport("
            f"health={self._overall_health.value}, "
            f"layers={self.n_layers}, "
            f"steps={self.completed_steps}, "
            f"dead={self.n_dead}, "
            f"cold={self.n_cold}"
            f")"
        )

    def __bool__(self) -> bool:
        """True when the report contains at least one layer with data."""
        return any(not s.is_empty for s in self._layer_stats.values())


# =============================================================================
# Section 5 — Private formatting helpers
# =============================================================================

def _shorten_layer_name(name: str, max_len: int = 45) -> str:
    """Shorten a fully-qualified module name for display.

    Preserves the tail of the name (most informative part).
    E.g. ``"model.layers.31.block_sparse_moe"`` → ``"...31.block_sparse_moe"``.
    """
    if len(name) <= max_len:
        return name
    return "..." + name[-(max_len - 3):]


def _word_wrap(text: str, width: int = 60, indent: str = "   ") -> str:
    """Wrap *text* to *width* characters with *indent* on continuation lines."""
    if len(text) <= width:
        return indent + text

    words  = text.split()
    lines  = []
    line   = indent
    is_first = True

    for word in words:
        if len(line) + len(word) + 1 > width and not is_first:
            lines.append(line.rstrip())
            line = indent + "  " + word + " "
        else:
            line += word + " "
            is_first = False

    if line.strip():
        lines.append(line.rstrip())

    return "\n".join(lines)
