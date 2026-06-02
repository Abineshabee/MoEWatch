# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  config.py — Central configuration for all moewatch diagnostics
#
#  All tunable thresholds, sampling rates, and output modes live here.
#  Every component receives a WatchConfig instance — there is no global
#  mutable state. Defaults are derived from empirical observations in the
#  Switch Transformer, Mixtral, and DeepSeek-MoE training literature.
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#  Version: 0.1.0
#
# =============================================================================

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List


# ---------------------------------------------------------------------------
# Output mode
# ---------------------------------------------------------------------------

class OutputMode(str, Enum):
    """Controls how moewatch emits diagnostic output.

    Attributes
    ----------
    CONSOLE:
        Human-readable coloured ASCII, written to stdout. Default.
    JSON:
        Structured newline-delimited JSON, suitable for log aggregators
        and programmatic consumption (e.g. Grafana, Splunk, custom pipelines).
    SILENT:
        Suppresses all real-time output; results only available via the
        returned ``AuditReport`` object.
    """

    CONSOLE = "console"
    JSON    = "json"
    SILENT  = "silent"


# ---------------------------------------------------------------------------
# Alert severity
# ---------------------------------------------------------------------------

class AlertLevel(str, Enum):
    """Severity levels used in live training-time alerts.

    Follows the standard INFO → WARN → ERROR escalation ladder. moewatch
    never emits FATAL — it diagnoses, it does not stop your training run.
    """

    INFO  = "INFO"
    WARN  = "WARN"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# WatchConfig
# ---------------------------------------------------------------------------

@dataclass
class WatchConfig:
    """Immutable configuration object for all moewatch components.

    Pass a single ``WatchConfig`` instance to ``audit()``, ``MoEWatch``, or
    any sub-component. All fields carry sensible defaults derived from the
    MoE training literature; override only what you need.

    Parameters
    ----------
    dead_threshold : float
        Fraction of total tokens below which an expert is considered *dead*.
        Default ``0.001`` (0.1 %).  Based on Switch Transformer findings where
        experts below 0.1 % utilisation never recovered during training.

    cold_threshold : float
        Fraction of total tokens below which an expert is considered *cold*
        (potentially recoverable). Must be > ``dead_threshold``.
        Default ``0.005`` (0.5 %).

    cold_steps_limit : int
        Number of consecutive steps an expert may remain cold before being
        promoted to *dead* status. Default ``500``.

    entropy_warn : float
        Fraction of theoretical maximum entropy below which a WARN is emitted.
        Default ``0.60`` (60 % of log₂(n_experts)).

    entropy_critical : float
        Fraction of theoretical maximum entropy below which an ERROR is emitted.
        Must be < ``entropy_warn``. Default ``0.40`` (40 %).

    entropy_drop_warn : float
        Relative entropy drop over ``window_steps`` that triggers a WARN.
        Default ``0.20`` (20 % relative drop).

    load_imbalance_warn : float
        Ratio of max expert utilisation to mean utilisation above which a WARN
        is emitted. Default ``3.0``.

    load_imbalance_error : float
        Same ratio, above which an ERROR is emitted. Must be > ``load_imbalance_warn``.
        Default ``5.0``.

    window_steps : int
        Rolling window size (in training steps) used for trend analysis and
        dead-expert duration tracking. Default ``500``.

    sample_every : int
        moewatch instruments every Nth forward pass to keep overhead below 2 %.
        Set to 1 to capture every step (maximum fidelity, ~2–4 % overhead).
        Default ``10``.

    log_every : int
        Emit a live alert summary every N training steps. Independent of
        ``sample_every``; alerts fire only when there is something to report.
        Default ``100``.

    ring_buffer_capacity : int
        Maximum number of ``RoutingEvent`` entries held in the in-memory ring
        buffer. Older events are overwritten when the buffer wraps. Tune this
        if you run very long training runs with large models — each event is
        ~(n_layers × n_experts × 4) bytes. Default ``10_000``.

    output : OutputMode or str
        Controls real-time diagnostic output. Accepts an ``OutputMode`` enum
        value or its string equivalent: ``"console"``, ``"json"``,
        ``"silent"``. Default ``OutputMode.CONSOLE``.

    router_modules : list of str, optional
        Explicit list of fully-qualified module names to instrument (e.g.
        ``["model.layers.0.block_sparse_moe"]``). When provided, auto-detection
        is bypassed entirely. Use this for custom MoE architectures that do not
        match any known naming pattern.

    no_color : bool
        Disable ANSI colour codes in console output. Respects the ``NO_COLOR``
        environment variable automatically; this flag lets you disable colour
        programmatically. Default ``False``.

    Examples
    --------
    Default config (recommended starting point):

    >>> config = WatchConfig()

    Aggressive thresholds for a large-expert model (64+ experts per layer):

    >>> config = WatchConfig(
    ...     dead_threshold=0.0001,
    ...     entropy_warn=0.50,
    ...     window_steps=1000,
    ... )

    Silent mode for integration with an existing logging pipeline:

    >>> config = WatchConfig(output=OutputMode.SILENT)

    Manual router override for a custom architecture:

    >>> config = WatchConfig(
    ...     router_modules=["model.moe_block.router", "model.moe_block2.router"]
    ... )
    """

    # -- Expert health thresholds -------------------------------------------
    dead_threshold:      float = 0.001   # 0.1 % utilisation → DEAD
    cold_threshold:      float = 0.005   # 0.5 % utilisation → COLD
    cold_steps_limit:    int   = 500     # steps cold before → DEAD

    # -- Entropy thresholds -------------------------------------------------
    entropy_warn:        float = 0.60    # < 60 % of H_max → WARN
    entropy_critical:    float = 0.40    # < 40 % of H_max → ERROR
    entropy_drop_warn:   float = 0.20    # 20 % relative drop → WARN

    # -- Load imbalance thresholds ------------------------------------------
    load_imbalance_warn:  float = 3.0   # max/mean > 3 → WARN
    load_imbalance_error: float = 5.0   # max/mean > 5 → ERROR

    # -- Sampling & windowing -----------------------------------------------
    window_steps:          int  = 500
    sample_every:          int  = 10
    log_every:             int  = 100
    ring_buffer_capacity:  int  = 10_000

    # -- Output & display ---------------------------------------------------
    output:   OutputMode = OutputMode.CONSOLE
    no_color: bool       = False

    # -- Router discovery ---------------------------------------------------
    router_modules: List[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Post-init validation
    # -----------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Defensive copy — prevents external list mutation from affecting config
        self.router_modules = list(self.router_modules)

        # Coerce string → OutputMode for ergonomic API
        if isinstance(self.output, str):
            try:
                self.output = OutputMode(self.output)
            except ValueError:
                valid = [m.value for m in OutputMode]
                raise ValueError(
                    f"[moewatch] Invalid output mode '{self.output}'. "
                    f"Must be one of: {valid}"
                )

        self._validate()

    def _validate(self) -> None:
        """Raise ``ValueError`` for any logically inconsistent configuration."""

        errors: list[str] = []

        # -- Range checks ---------------------------------------------------
        def _check_fraction(name: str, value: float) -> None:
            if not (0.0 < value < 1.0):
                errors.append(
                    f"  {name}={value!r} must be in the open interval (0, 1)."
                )

        _check_fraction("dead_threshold",    self.dead_threshold)
        _check_fraction("cold_threshold",    self.cold_threshold)
        _check_fraction("entropy_warn",      self.entropy_warn)
        _check_fraction("entropy_critical",  self.entropy_critical)
        _check_fraction("entropy_drop_warn", self.entropy_drop_warn)

        # -- Ordering constraints -------------------------------------------
        if self.dead_threshold >= self.cold_threshold:
            errors.append(
                f"  dead_threshold ({self.dead_threshold}) must be strictly "
                f"less than cold_threshold ({self.cold_threshold})."
            )

        if self.entropy_critical >= self.entropy_warn:
            errors.append(
                f"  entropy_critical ({self.entropy_critical}) must be strictly "
                f"less than entropy_warn ({self.entropy_warn})."
            )

        if self.load_imbalance_warn >= self.load_imbalance_error:
            errors.append(
                f"  load_imbalance_warn ({self.load_imbalance_warn}) must be "
                f"strictly less than load_imbalance_error ({self.load_imbalance_error})."
            )

        # -- Positive integer checks ----------------------------------------
        for name, value in [
            ("window_steps",         self.window_steps),
            ("sample_every",         self.sample_every),
            ("log_every",            self.log_every),
            ("ring_buffer_capacity", self.ring_buffer_capacity),
            ("cold_steps_limit",     self.cold_steps_limit),
        ]:
            if not isinstance(value, int) or value < 1:
                errors.append(f"  {name}={value!r} must be a positive integer.")

        # -- Soft warnings (non-fatal) --------------------------------------
        if self.sample_every == 1:
            warnings.warn(
                "[moewatch] sample_every=1 captures every forward pass. "
                "Overhead may exceed 2 % on large models. Consider sample_every=5 "
                "or higher for production training runs.",
                UserWarning,
                stacklevel=3,
            )

        if self.ring_buffer_capacity < self.window_steps:
            warnings.warn(
                f"[moewatch] ring_buffer_capacity ({self.ring_buffer_capacity}) "
                f"is smaller than window_steps ({self.window_steps}). "
                "Trend analysis will operate on a truncated history. "
                "Consider increasing ring_buffer_capacity.",
                UserWarning,
                stacklevel=3,
            )

        # -- Raise aggregated errors ----------------------------------------
        if errors:
            raise ValueError(
                "[moewatch] WatchConfig validation failed:\n"
                + "\n".join(errors)
            )

    # -----------------------------------------------------------------------
    # Convenience constructors
    # -----------------------------------------------------------------------

    @classmethod
    def default(cls) -> "WatchConfig":
        """Return a WatchConfig with all defaults. Equivalent to ``WatchConfig()``."""
        return cls()

    @classmethod
    def aggressive(cls) -> "WatchConfig":
        """Tighter thresholds, faster sampling — use for debugging collapse.

        Suitable for short experimental runs where you want maximum signal.
        *Not* recommended for production training (overhead ~4–8 %).
        """
        return cls(
            dead_threshold=0.0005,
            cold_threshold=0.002,
            entropy_warn=0.70,
            entropy_critical=0.50,
            entropy_drop_warn=0.10,
            sample_every=1,
            log_every=50,
            window_steps=200,
        )

    @classmethod
    def lightweight(cls) -> "WatchConfig":
        """Minimal overhead — ideal for large-scale production training runs.

        Samples every 50 steps, logs every 500 steps, uses a compact buffer.
        """
        return cls(
            sample_every=50,
            log_every=500,
            ring_buffer_capacity=2_000,
        )

    @classmethod
    def silent(cls) -> "WatchConfig":
        """No real-time output. Results only available via the AuditReport."""
        return cls(output=OutputMode.SILENT)

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "dead_threshold":        self.dead_threshold,
            "cold_threshold":        self.cold_threshold,
            "cold_steps_limit":      self.cold_steps_limit,
            "entropy_warn":          self.entropy_warn,
            "entropy_critical":      self.entropy_critical,
            "entropy_drop_warn":     self.entropy_drop_warn,
            "load_imbalance_warn":   self.load_imbalance_warn,
            "load_imbalance_error":  self.load_imbalance_error,
            "window_steps":          self.window_steps,
            "sample_every":          self.sample_every,
            "log_every":             self.log_every,
            "ring_buffer_capacity":  self.ring_buffer_capacity,
            "output":                self.output.value,
            "no_color":              self.no_color,
            "router_modules":        list(self.router_modules),
        }

    def __repr__(self) -> str:  # pragma: no cover
        lines = ["WatchConfig("]
        for key, value in self.to_dict().items():
            lines.append(f"    {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)
