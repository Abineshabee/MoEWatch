# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  analyzer/entropy.py — Per-layer Shannon routing-entropy computation
#
#  Routing entropy is the primary early-warning signal for expert collapse.
#  A healthy MoE router distributes tokens near-uniformly across experts;
#  entropy close to log₂(n_experts) (theoretical maximum) indicates healthy
#  routing diversity. A sharp entropy drop — even before any expert reaches
#  zero utilisation — is the canonical precursor to full collapse.
#
#  This module provides:
#
#    compute_entropy(probs)
#        Pure function. Shannon H = -Σ pᵢ log₂(pᵢ) for a probability vector.
#        Numerically safe: zero probabilities contribute zero (0·log(0) ≡ 0).
#
#    EntropyResult
#        Immutable dataclass holding all entropy metrics for one layer at one
#        point in time: absolute entropy, normalised entropy, alert level,
#        and trend direction.
#
#    LayerEntropyReport
#        Thin container grouping EntropyResult objects for all layers, plus
#        convenience accessors (worst layer, layers below threshold).
#
#    EntropyAnalyzer
#        Stateful analyser. Consumes a Dict[str, LayerStats] from the
#        StatCollector and emits a LayerEntropyReport. Tracks per-layer
#        entropy history to detect monotonic drops (trend analysis).
#
#  Computation sources (priority order)
#  -------------------------------------
#  1. Raw logits  — if LayerStats.has_raw_logits is True, entropy is computed
#                   from the softmax distribution over raw logit tensors.
#                   This is more precise than counts because it captures the
#                   full router distribution, not just the top-k selection.
#  2. Token counts — when raw logits are absent, entropy is estimated from
#                    the empirical token-count distribution (utilization tensor).
#                    This is always available and accurate for large batches.
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#  Version: 0.1.0
#
# =============================================================================

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import torch

from ..config import AlertLevel, WatchConfig
from ..collector.stat_collector import LayerStats

log = logging.getLogger(__name__)


# =============================================================================
# Section 1 — Pure functional core
# =============================================================================

# ------------------------------------------------------------------------------
# §1.1  Numerical constants
# ------------------------------------------------------------------------------

#: Minimum probability mass to treat as non-zero. Values below this are
#: clamped to zero before entropy computation to avoid log(ε) artifacts.
_PROB_EPS: float = 1e-9

#: Natural log of 2 — used to convert nats → bits when computing in log₂.
_LN2: float = math.log(2.0)


# ------------------------------------------------------------------------------
# §1.2  compute_entropy — Shannon entropy of a probability vector
# ------------------------------------------------------------------------------

def compute_entropy(probs: torch.Tensor) -> float:
    """Compute Shannon entropy H = -Σ pᵢ log₂(pᵢ) for a probability vector.

    Numerically safe:
      - Zero probabilities are masked out (0 · log(0) ≡ 0 by convention).
      - The input is re-normalised if it does not sum to 1.0 within tolerance.
      - Works for any number of categories ≥ 1.

    Parameters
    ----------
    probs : torch.Tensor
        1-D float tensor of shape ``(n_experts,)`` representing a probability
        distribution. Values must be non-negative; need not sum to exactly 1.0
        (re-normalised internally). The tensor may live on any device; it is
        moved to CPU before computation.

    Returns
    -------
    float
        Shannon entropy in **bits** (log base 2). Range: [0.0, log₂(n_experts)].
        Returns ``0.0`` for degenerate inputs (all-zero, single-element, or
        one-hot distributions).

    Raises
    ------
    ValueError
        If ``probs`` is not 1-D, or if it contains negative values.

    Examples
    --------
    >>> import torch
    >>> compute_entropy(torch.tensor([0.5, 0.5]))          # 1.0 bit
    1.0
    >>> compute_entropy(torch.tensor([1.0, 0.0, 0.0]))    # 0.0 bits (one-hot)
    0.0
    >>> compute_entropy(torch.ones(8) / 8)                 # 3.0 bits (uniform 8)
    3.0
    """
    if probs.ndim != 1:
        raise ValueError(
            f"compute_entropy expects a 1-D tensor, got shape {list(probs.shape)}."
        )

    p = probs.detach().cpu().float()

    if (p < 0.0).any():
        raise ValueError(
            "compute_entropy: probability vector contains negative values."
        )

    total = p.sum().item()
    if total <= 0.0:
        # All-zero vector — entropy is undefined; return 0.0 by convention.
        return 0.0

    # Re-normalise to a valid probability distribution.
    if abs(total - 1.0) > 1e-4:
        p = p / total

    # Mask out near-zero probabilities (0 · log(0) ≡ 0).
    mask = p > _PROB_EPS
    if not mask.any():
        return 0.0

    p_nz = p[mask]
    # Shannon entropy in bits: H = -Σ pᵢ log₂(pᵢ)
    entropy = -(p_nz * p_nz.log()).sum().item() / _LN2

    # Clamp to [0, log₂(n)] to absorb floating-point rounding.
    n_experts = probs.shape[0]
    h_max     = math.log2(n_experts) if n_experts > 1 else 0.0
    return float(max(0.0, min(entropy, h_max)))


def compute_entropy_from_logits(logits: torch.Tensor) -> float:
    """Compute routing entropy directly from raw router logit tensors.

    Applies softmax row-wise to obtain per-token routing distributions, then
    averages entropy across all tokens. This is more accurate than count-based
    estimation for small batch sizes.

    Parameters
    ----------
    logits : torch.Tensor
        2-D float tensor of shape ``(total_tokens, n_experts)`` — the raw
        (un-softmaxed) router output. Must be 2-D; a single token should be
        unsqueezed to ``(1, n_experts)`` before calling.

    Returns
    -------
    float
        Mean per-token Shannon entropy in bits. Range: [0.0, log₂(n_experts)].

    Raises
    ------
    ValueError
        If ``logits`` is not 2-D or has fewer than 1 expert.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"compute_entropy_from_logits expects 2-D logits, "
            f"got shape {list(logits.shape)}."
        )

    total_tokens, n_experts = logits.shape
    if n_experts < 1:
        return 0.0
    if total_tokens == 0:
        return 0.0

    with torch.no_grad():
        log_probs = torch.log_softmax(logits.float().cpu(), dim=-1)  # (T, E)
        probs     = log_probs.exp()                                   # (T, E)
        # Per-token entropy: -Σ p log₂(p) = -Σ p log(p) / ln(2)
        per_token_entropy = -(probs * log_probs).sum(dim=-1) / _LN2  # (T,)
        mean_entropy      = per_token_entropy.mean().item()

    h_max = math.log2(n_experts) if n_experts > 1 else 0.0
    return float(max(0.0, min(mean_entropy, h_max)))


def max_entropy(n_experts: int) -> float:
    """Theoretical maximum Shannon entropy for *n_experts* (uniform distribution).

    Parameters
    ----------
    n_experts : int
        Number of experts. Must be ≥ 1.

    Returns
    -------
    float
        log₂(n_experts). Returns 0.0 for n_experts ≤ 1.
    """
    if n_experts <= 1:
        return 0.0
    return math.log2(n_experts)


def normalised_entropy(h: float, n_experts: int) -> float:
    """Normalise absolute entropy to [0.0, 1.0] relative to the theoretical maximum.

    Parameters
    ----------
    h : float
        Absolute entropy in bits.
    n_experts : int
        Total number of experts (determines H_max).

    Returns
    -------
    float
        ``h / log₂(n_experts)``, clamped to [0.0, 1.0].
        Returns 0.0 when n_experts ≤ 1 or H_max == 0.
    """
    h_max = max_entropy(n_experts)
    if h_max <= 0.0:
        return 0.0
    return float(max(0.0, min(h / h_max, 1.0)))


# =============================================================================
# Section 2 — Data containers
# =============================================================================

# ------------------------------------------------------------------------------
# §2.1  TrendDirection — direction of entropy change over the recent window
# ------------------------------------------------------------------------------

class TrendDirection:
    """Symbolic trend direction constants (not an Enum for easy string-ability)."""

    STABLE:    str = "STABLE"     # |ΔH| < entropy_drop_warn threshold
    IMPROVING: str = "IMPROVING"  # entropy is increasing (healthy recovery)
    DECLINING: str = "DECLINING"  # entropy is decreasing (collapse precursor) — WARN
    UNKNOWN:   str = "UNKNOWN"    # fewer than 2 history points, no trend yet


# ------------------------------------------------------------------------------
# §2.2  EntropyResult — per-layer entropy snapshot
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class EntropyResult:
    """Immutable entropy snapshot for a single MoE router layer.

    Produced by :class:`EntropyAnalyzer` and stored in
    :class:`LayerEntropyReport`.

    Attributes
    ----------
    layer_name : str
        Fully-qualified module name.
    n_experts : int
        Number of experts in this layer.
    entropy_bits : float
        Absolute Shannon entropy in bits.  Range: [0, log₂(n_experts)].
    entropy_norm : float
        Entropy normalised to [0, 1] relative to H_max = log₂(n_experts).
        1.0 = perfectly uniform; 0.0 = fully collapsed (one-hot routing).
    h_max : float
        Theoretical maximum entropy (log₂(n_experts)).
    alert_level : AlertLevel
        Severity of the entropy alert (INFO / WARN / ERROR).
    trend : str
        One of TrendDirection constants — direction of entropy change.
    trend_delta : float
        Relative entropy change over the history window: ``(H_now - H_prev) / H_prev``.
        Positive = improving; negative = declining.  ``0.0`` when unknown.
    source : str
        ``"logits"`` when computed from raw router logits (more accurate),
        ``"counts"`` when estimated from token-count distribution.
    event_count : int
        Number of RoutingEvent objects that contributed to this result.
    is_empty : bool
        True when no events were available (layer has never fired).
    """

    layer_name:   str
    n_experts:    int
    entropy_bits: float
    entropy_norm: float
    h_max:        float
    alert_level:  AlertLevel
    trend:        str
    trend_delta:  float
    source:       str         # "logits" | "counts"
    event_count:  int
    is_empty:     bool

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """True when alert_level is INFO (no threshold breached)."""
        return self.alert_level == AlertLevel.INFO

    @property
    def is_critical(self) -> bool:
        """True when entropy is in the ERROR zone."""
        return self.alert_level == AlertLevel.ERROR

    @property
    def is_declining(self) -> bool:
        """True when the entropy trend is DECLINING (collapse precursor)."""
        return self.trend == TrendDirection.DECLINING

    def to_dict(self) -> dict:
        """JSON-serialisable dictionary representation."""
        return {
            "layer_name":   self.layer_name,
            "n_experts":    self.n_experts,
            "entropy_bits": round(self.entropy_bits, 6),
            "entropy_norm": round(self.entropy_norm, 6),
            "h_max":        round(self.h_max, 6),
            "alert_level":  self.alert_level.value,
            "trend":        self.trend,
            "trend_delta":  round(self.trend_delta, 6),
            "source":       self.source,
            "event_count":  self.event_count,
            "is_empty":     self.is_empty,
        }

    def __str__(self) -> str:
        if self.is_empty:
            return f"EntropyResult({self.layer_name!r}: NO DATA)"
        return (
            f"EntropyResult({self.layer_name!r}: "
            f"H={self.entropy_bits:.3f} bits "
            f"[{self.entropy_norm * 100:.1f}% of max], "
            f"level={self.alert_level.value}, "
            f"trend={self.trend})"
        )


# ------------------------------------------------------------------------------
# §2.3  LayerEntropyReport — collection of EntropyResult for all layers
# ------------------------------------------------------------------------------

@dataclass
class LayerEntropyReport:
    """Entropy report for all tracked MoE layers.

    Returned by :meth:`EntropyAnalyzer.analyze` and consumed by
    :class:`~moewatch.report.audit_report.AuditReport`.

    Attributes
    ----------
    results : dict
        ``{layer_name: EntropyResult}`` — ordered by layer name.
    global_alert : AlertLevel
        Worst alert level across all layers (INFO < WARN < ERROR).
    n_warn : int
        Number of layers in WARN state.
    n_error : int
        Number of layers in ERROR state.
    n_declining : int
        Number of layers with DECLINING entropy trend.
    """

    results:      Dict[str, EntropyResult] = field(default_factory=dict)
    global_alert: AlertLevel               = AlertLevel.INFO
    n_warn:       int                      = 0
    n_error:      int                      = 0
    n_declining:  int                      = 0

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def layers(self) -> List[str]:
        """Ordered list of layer names in the report."""
        return list(self.results.keys())

    @property
    def is_healthy(self) -> bool:
        """True when every layer is in the INFO state."""
        return self.global_alert == AlertLevel.INFO

    def worst_layer(self) -> Optional[EntropyResult]:
        """Return the EntropyResult with the lowest normalised entropy."""
        non_empty = [r for r in self.results.values() if not r.is_empty]
        if not non_empty:
            return None
        return min(non_empty, key=lambda r: r.entropy_norm)

    def layers_below(self, threshold_norm: float) -> List[EntropyResult]:
        """Return all non-empty layers whose normalised entropy is below *threshold_norm*.

        Parameters
        ----------
        threshold_norm : float
            Fraction of H_max.  E.g. ``0.6`` means below 60 % of maximum entropy.
        """
        return [
            r for r in self.results.values()
            if not r.is_empty and r.entropy_norm < threshold_norm
        ]

    def declining_layers(self) -> List[EntropyResult]:
        """Return all layers with a DECLINING entropy trend."""
        return [
            r for r in self.results.values()
            if r.trend == TrendDirection.DECLINING
        ]

    def to_dict(self) -> dict:
        """JSON-serialisable representation of the full report."""
        return {
            "global_alert": self.global_alert.value,
            "n_warn":       self.n_warn,
            "n_error":      self.n_error,
            "n_declining":  self.n_declining,
            "layers":       {name: r.to_dict() for name, r in self.results.items()},
        }

    def __repr__(self) -> str:
        return (
            f"LayerEntropyReport("
            f"layers={len(self.results)}, "
            f"global_alert={self.global_alert.value}, "
            f"n_warn={self.n_warn}, "
            f"n_error={self.n_error}, "
            f"n_declining={self.n_declining}"
            f")"
        )


# =============================================================================
# Section 3 — EntropyAnalyzer
# =============================================================================

class EntropyAnalyzer:
    """Stateful per-layer Shannon entropy analyser for MoE routing diagnostics.

    Consumes :class:`~moewatch.collector.stat_collector.LayerStats` snapshots
    produced by the :class:`~moewatch.collector.stat_collector.StatCollector`
    and emits a :class:`LayerEntropyReport` containing entropy values, alert
    levels, and trend direction for every tracked layer.

    Statefulness
    ------------
    ``EntropyAnalyzer`` maintains a per-layer entropy history (a fixed-length
    deque of recent entropy values). This history is used for trend detection:
    if the mean entropy over the last ``_TREND_WINDOW`` calls has dropped by
    more than ``config.entropy_drop_warn`` relative to the mean over the prior
    window, a DECLINING trend is flagged.

    The analyser is intentionally stateful so that a single instance can be
    shared by both the offline ``audit()`` path and the live ``MoEWatch``
    training-time path without losing trend history.

    Thread safety
    -------------
    ``analyze()`` is not thread-safe. Call it from a single thread (the
    analyzer/reporter thread); hook callbacks run on a separate thread but
    write only to the StatCollector, not to this class.

    Parameters
    ----------
    config : WatchConfig
        Provides ``entropy_warn``, ``entropy_critical``, and
        ``entropy_drop_warn`` thresholds.

    Examples
    --------
    >>> analyzer = EntropyAnalyzer(config)
    >>> stats = collector.get_all_stats()
    >>> report = analyzer.analyze(stats)
    >>> report.global_alert
    <AlertLevel.INFO: 'INFO'>
    >>> report.worst_layer()
    EntropyResult('model.layers.7.block_sparse_moe': H=1.23 bits [41.0% of max], ...)
    """

    # Number of historical entropy values kept per layer for trend detection.
    _HISTORY_MAXLEN: int = 50

    # Minimum history points required before a trend verdict is emitted.
    _TREND_MIN_POINTS: int = 4

    # Half the history used for "recent" vs "prior" in trend comparison.
    # Recent = last _TREND_SPLIT points; Prior = the _TREND_SPLIT before that.
    _TREND_SPLIT: int = 5

    def __init__(self, config: WatchConfig) -> None:
        self.config = config

        # Per-layer deque of (entropy_norm,) values — oldest at left.
        self._history: Dict[str, Deque[float]] = {}

        log.debug(
            "[moewatch] EntropyAnalyzer initialised "
            "(entropy_warn=%.2f, entropy_critical=%.2f, entropy_drop_warn=%.2f).",
            config.entropy_warn,
            config.entropy_critical,
            config.entropy_drop_warn,
        )

    # ==========================================================================
    # §3.1  Public interface
    # ==========================================================================

    def analyze(
        self,
        all_stats: Dict[str, LayerStats],
    ) -> LayerEntropyReport:
        """Run entropy analysis over all tracked layers.

        Computes an :class:`EntropyResult` for every layer in *all_stats*,
        updates internal history, and assembles a :class:`LayerEntropyReport`.

        Parameters
        ----------
        all_stats : dict
            ``{layer_name: LayerStats}`` as returned by
            :meth:`~moewatch.collector.stat_collector.StatCollector.get_all_stats`.
            An empty dict is valid and produces an empty report.

        Returns
        -------
        LayerEntropyReport
            Report with per-layer entropy results and summary-level counters.
        """
        if not all_stats:
            log.debug("[moewatch] EntropyAnalyzer.analyze() called with empty stats.")
            return LayerEntropyReport()

        results: Dict[str, EntropyResult] = {}
        n_warn    = 0
        n_error   = 0
        n_decline = 0
        worst_level = AlertLevel.INFO

        for layer_name, stats in all_stats.items():
            result = self._analyze_layer(layer_name, stats)
            results[layer_name] = result

            # Aggregate counters
            if result.alert_level == AlertLevel.WARN:
                n_warn    += 1
            elif result.alert_level == AlertLevel.ERROR:
                n_error   += 1
                n_warn    += 1  # ERROR is a superset of WARN for reporting

            if result.trend == TrendDirection.DECLINING:
                n_decline += 1

            # Track worst level
            if result.alert_level == AlertLevel.ERROR:
                worst_level = AlertLevel.ERROR
            elif result.alert_level == AlertLevel.WARN and worst_level == AlertLevel.INFO:
                worst_level = AlertLevel.WARN

        report = LayerEntropyReport(
            results=results,
            global_alert=worst_level,
            n_warn=n_warn,
            n_error=n_error,
            n_declining=n_decline,
        )

        log.debug(
            "[moewatch] EntropyAnalyzer: %d layer(s) analysed — "
            "global_alert=%s, n_warn=%d, n_error=%d, n_declining=%d.",
            len(results),
            worst_level.value,
            n_warn,
            n_error,
            n_decline,
        )

        return report

    def reset_history(self, layer_name: Optional[str] = None) -> None:
        """Clear entropy history for a specific layer or all layers.

        Useful when a model is re-initialised mid-run (e.g. checkpoint reload).

        Parameters
        ----------
        layer_name : str, optional
            If provided, clear history only for that layer.
            If ``None``, clear all history.
        """
        if layer_name is not None:
            self._history.pop(layer_name, None)
            log.debug("[moewatch] EntropyAnalyzer: history cleared for %s.", layer_name)
        else:
            self._history.clear()
            log.debug("[moewatch] EntropyAnalyzer: all history cleared.")

    @property
    def tracked_layers(self) -> List[str]:
        """Names of all layers that have at least one history point."""
        return list(self._history.keys())

    def history_for(self, layer_name: str) -> List[float]:
        """Return the normalised entropy history list for a layer (oldest first).

        Returns an empty list if the layer has never been analysed.
        """
        return list(self._history.get(layer_name, []))

    # ==========================================================================
    # §3.2  Per-layer analysis
    # ==========================================================================

    def _analyze_layer(
        self,
        layer_name: str,
        stats: LayerStats,
    ) -> EntropyResult:
        """Compute :class:`EntropyResult` for a single layer."""

        # -- Edge case: no data yet -------------------------------------------
        if stats.is_empty or stats.n_experts == 0:
            return self._empty_result(layer_name, stats.n_experts)

        # -- Step 1: compute absolute entropy ---------------------------------
        h_max  = max_entropy(stats.n_experts)
        source = "counts"

        if stats.has_raw_logits and stats.raw_logits_window:
            # Preferred: compute from raw logits when available.
            h_bits, source = self._entropy_from_logits(
                stats.raw_logits_window, stats.n_experts, h_max
            )
        else:
            # Fallback: estimate from token-count distribution.
            h_bits = compute_entropy(stats.utilization)
            source = "counts"

        h_norm = normalised_entropy(h_bits, stats.n_experts)

        # -- Step 2: determine alert level ------------------------------------
        alert_level = self._classify_entropy(h_norm)

        # -- Step 3: update history and determine trend -----------------------
        self._push_history(layer_name, h_norm)
        trend, trend_delta = self._compute_trend(layer_name)

        # A DECLINING trend at WARN level may upgrade alert to WARN
        # even if entropy is momentarily above the warn threshold.
        if (
            alert_level == AlertLevel.INFO
            and trend == TrendDirection.DECLINING
            and abs(trend_delta) >= self.config.entropy_drop_warn
        ):
            alert_level = AlertLevel.WARN
            log.debug(
                "[moewatch] EntropyAnalyzer(%s): alert upgraded to WARN "
                "due to DECLINING trend (delta=%.3f).",
                layer_name,
                trend_delta,
            )

        # -- Step 4: log notable findings ------------------------------------
        self._log_alert(layer_name, h_bits, h_norm, h_max, alert_level, trend)

        return EntropyResult(
            layer_name=layer_name,
            n_experts=stats.n_experts,
            entropy_bits=h_bits,
            entropy_norm=h_norm,
            h_max=h_max,
            alert_level=alert_level,
            trend=trend,
            trend_delta=trend_delta,
            source=source,
            event_count=stats.event_count,
            is_empty=False,
        )

    # ==========================================================================
    # §3.3  Entropy computation helpers
    # ==========================================================================

    def _entropy_from_logits(
        self,
        logits_window: List[torch.Tensor],
        n_experts: int,
        h_max: float,
    ) -> Tuple[float, str]:
        """Compute mean entropy from a list of logit tensors.

        Each tensor has shape ``(total_tokens_in_batch, n_experts)``.
        We concatenate all tensors and compute a single entropy over the
        pooled distribution. Tensors with mismatched expert dimensions are
        silently skipped.

        Returns
        -------
        (entropy_bits, source)
        """
        valid = [
            t for t in logits_window
            if isinstance(t, torch.Tensor)
            and t.ndim == 2
            and t.shape[1] == n_experts
            and t.shape[0] > 0
        ]

        if not valid:
            # Nothing usable — fall back to count-based (handled by caller)
            return 0.0, "counts"

        try:
            pooled   = torch.cat(valid, dim=0)           # (total_tokens, n_experts)
            h_bits   = compute_entropy_from_logits(pooled)
            return h_bits, "logits"
        except Exception as exc:
            log.warning(
                "[moewatch] EntropyAnalyzer: logit-based entropy computation "
                "failed (%s). Falling back to count-based.", exc
            )
            return 0.0, "counts"

    # ==========================================================================
    # §3.4  Alert classification
    # ==========================================================================

    def _classify_entropy(self, h_norm: float) -> AlertLevel:
        """Map normalised entropy to an AlertLevel.

        Thresholds (from WatchConfig, fraction of H_max):
          - h_norm < entropy_critical → ERROR
          - h_norm < entropy_warn     → WARN
          - otherwise                 → INFO
        """
        if h_norm < self.config.entropy_critical:
            return AlertLevel.ERROR
        if h_norm < self.config.entropy_warn:
            return AlertLevel.WARN
        return AlertLevel.INFO

    # ==========================================================================
    # §3.5  Trend analysis
    # ==========================================================================

    def _push_history(self, layer_name: str, h_norm: float) -> None:
        """Append *h_norm* to the per-layer entropy history deque."""
        if layer_name not in self._history:
            self._history[layer_name] = deque(maxlen=self._HISTORY_MAXLEN)
        self._history[layer_name].append(h_norm)

    def _compute_trend(
        self,
        layer_name: str,
    ) -> Tuple[str, float]:
        """Determine entropy trend direction for *layer_name*.

        Strategy
        --------
        Split the history into two halves (recent vs prior). If the mean of
        the recent half is lower than the mean of the prior half by more than
        ``config.entropy_drop_warn``, flag as DECLINING. If higher, IMPROVING.
        Otherwise, STABLE.

        Returns
        -------
        (trend_direction, trend_delta)
            trend_delta = (recent_mean - prior_mean) / prior_mean
            Positive = entropy increased (good); negative = decreased (bad).
        """
        hist = self._history.get(layer_name)
        if hist is None or len(hist) < self._TREND_MIN_POINTS:
            return TrendDirection.UNKNOWN, 0.0

        history_list = list(hist)
        n            = len(history_list)
        split        = max(1, min(self._TREND_SPLIT, n // 2))

        recent_vals = history_list[-split:]
        prior_vals  = history_list[-(split * 2):-split] if n >= split * 2 else history_list[:split]

        if not prior_vals:
            return TrendDirection.UNKNOWN, 0.0

        recent_mean = sum(recent_vals) / len(recent_vals)
        prior_mean  = sum(prior_vals)  / len(prior_vals)

        if prior_mean <= 0.0:
            return TrendDirection.UNKNOWN, 0.0

        delta = (recent_mean - prior_mean) / prior_mean   # relative change

        if delta < -self.config.entropy_drop_warn:
            return TrendDirection.DECLINING, delta
        if delta > self.config.entropy_drop_warn:
            return TrendDirection.IMPROVING, delta
        return TrendDirection.STABLE, delta

    # ==========================================================================
    # §3.6  Logging helpers
    # ==========================================================================

    def _log_alert(
        self,
        layer_name: str,
        h_bits:     float,
        h_norm:     float,
        h_max:      float,
        alert:      AlertLevel,
        trend:      str,
    ) -> None:
        """Emit a structured log line for notable entropy conditions."""
        if alert == AlertLevel.ERROR:
            log.error(
                "[moewatch] ENTROPY CRITICAL  %-50s  "
                "H=%.3f / %.3f bits  (%.1f%% of max)  trend=%s",
                layer_name, h_bits, h_max, h_norm * 100, trend,
            )
        elif alert == AlertLevel.WARN:
            log.warning(
                "[moewatch] ENTROPY LOW       %-50s  "
                "H=%.3f / %.3f bits  (%.1f%% of max)  trend=%s",
                layer_name, h_bits, h_max, h_norm * 100, trend,
            )
        else:
            log.debug(
                "[moewatch] entropy OK        %-50s  "
                "H=%.3f / %.3f bits  (%.1f%% of max)  trend=%s",
                layer_name, h_bits, h_max, h_norm * 100, trend,
            )

    # ==========================================================================
    # §3.7  Edge-case helpers
    # ==========================================================================

    @staticmethod
    def _empty_result(layer_name: str, n_experts: int) -> EntropyResult:
        """Return a sentinel EntropyResult for a layer with no data."""
        return EntropyResult(
            layer_name=layer_name,
            n_experts=n_experts,
            entropy_bits=0.0,
            entropy_norm=0.0,
            h_max=max_entropy(n_experts),
            alert_level=AlertLevel.INFO,   # not an error — just no data yet
            trend=TrendDirection.UNKNOWN,
            trend_delta=0.0,
            source="none",
            event_count=0,
            is_empty=True,
        )

    # ==========================================================================
    # §3.8  Dunder methods
    # ==========================================================================

    def __repr__(self) -> str:
        return (
            f"EntropyAnalyzer("
            f"tracked_layers={len(self._history)}, "
            f"entropy_warn={self.config.entropy_warn}, "
            f"entropy_critical={self.config.entropy_critical}"
            f")"
        )
