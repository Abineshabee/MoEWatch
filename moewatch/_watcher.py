# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  _watcher.py — Live training-time MoE diagnostic monitor
#
#  MoEWatch sits beside your training loop and emits structured alerts
#  whenever routing health degrades. It integrates with the HuggingFace
#  Trainer via a standard TrainerCallback, and also supports arbitrary
#  custom training loops via the standalone step() method.
#
#  Lifecycle
#  ---------
#    watcher = MoEWatch(model, config=config)
#    watcher.attach(trainer)   # HuggingFace Trainer path
#    # ... training runs ...
#    watcher.detach()          # clean hook removal
#
#  Custom loop
#  -----------
#    watcher = MoEWatch(model)
#    watcher.start()
#    for step, batch in enumerate(dataloader):
#        loss = model(**batch).loss
#        loss.backward()
#        optimizer.step()
#        watcher.step(step)        # check health every N steps
#    watcher.stop()
#
#  Design principles
#  -----------------
#    - Zero weight modifications at all times.
#    - Thread-safe alert emission (GIL is sufficient for Python logging).
#    - Always detach on exception — no leaked PyTorch hooks.
#    - Alert history preserved for post-training audit via get_alert_log().
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#  Version: 0.1.0
#
# =============================================================================

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .config import AlertLevel, OutputMode, WatchConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert data structure
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """A single diagnostic alert emitted during training.

    Attributes
    ----------
    step       : Training step at which the alert fired.
    level      : Severity — INFO, WARN, or ERROR.
    layer_name : Fully-qualified name of the affected router module.
    message    : Human-readable description of the condition.
    metric     : The numeric value that triggered the alert (e.g. utilisation).
    suggestion : Actionable fix suggestion for ERROR-level alerts.
    timestamp  : Wall-clock time (seconds since epoch) of the alert.
    """

    step:        int
    level:       AlertLevel
    layer_name:  str
    message:     str
    metric:      Optional[float]        = None
    suggestion:  Optional[str]          = None
    timestamp:   float                  = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":        self.step,
            "level":       self.level.value,
            "layer_name":  self.layer_name,
            "message":     self.message,
            "metric":      self.metric,
            "suggestion":  self.suggestion,
            "timestamp":   self.timestamp,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# ANSI colour helpers (respect NO_COLOR env var and WatchConfig.no_color)
# ---------------------------------------------------------------------------

_ANSI_RESET  = "\033[0m"
_ANSI_BOLD   = "\033[1m"
_ANSI_GREEN  = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RED    = "\033[31m"
_ANSI_CYAN   = "\033[36m"

_LEVEL_COLOUR = {
    AlertLevel.INFO:  _ANSI_GREEN,
    AlertLevel.WARN:  _ANSI_YELLOW,
    AlertLevel.ERROR: _ANSI_RED,
}

_LEVEL_ICON = {
    AlertLevel.INFO:  "✅",
    AlertLevel.WARN:  "⚠ ",
    AlertLevel.ERROR: "❌",
}


def _colour(text: str, code: str, *, no_color: bool) -> str:
    if no_color:
        return text
    return f"{code}{text}{_ANSI_RESET}"


# ---------------------------------------------------------------------------
# Suggestion database
# ---------------------------------------------------------------------------

_SUGGESTIONS: Dict[str, str] = {
    "dead_expert": (
        "Consider raising aux_loss_coef (e.g. 0.02 → 0.04) to strengthen "
        "load-balancing pressure. If using Mixtral, check router_aux_loss_coef "
        "in the model config. Alternatively, lower the learning rate for the "
        "router parameters specifically."
    ),
    "low_entropy": (
        "Routing entropy has dropped significantly. Possible causes: (1) the "
        "auxiliary loss coefficient is too low, (2) the router learning rate "
        "is too high relative to expert learning rates, or (3) a gradient "
        "explosion has destabilised the router. Check your loss curve for "
        "spikes at this step."
    ),
    "load_imbalance": (
        "Expert load is highly skewed. If using token-choice routing (top-k), "
        "consider adding a capacity factor or switching to expert-choice "
        "routing. For Mixtral-style models, verify that router_jitter_noise > 0 "
        "in the model config."
    ),
}


# ---------------------------------------------------------------------------
# MoEWatch — main class
# ---------------------------------------------------------------------------

class MoEWatch:
    """Live training-time MoE diagnostic monitor.

    Attaches PyTorch forward hooks to router modules and emits structured
    alerts whenever routing health degrades. Works with HuggingFace Trainer
    (via ``attach(trainer)``) or any custom training loop (via ``step()``).

    Parameters
    ----------
    model : torch.nn.Module
        The MoE model to monitor. Never modified.
    config : WatchConfig, optional
        Diagnostic configuration. Defaults to ``WatchConfig()`` if not given.

    Examples
    --------
    HuggingFace Trainer integration:

    >>> watcher = MoEWatch(model)
    >>> watcher.attach(trainer)  # injects MoEWatchCallback automatically
    >>> trainer.train()
    >>> watcher.detach()

    Custom training loop:

    >>> watcher = MoEWatch(model, config=WatchConfig(log_every=50))
    >>> watcher.start()
    >>> for step, batch in enumerate(dataloader):
    ...     loss = model(**batch).loss
    ...     loss.backward()
    ...     optimizer.step()
    ...     watcher.step(step)
    >>> watcher.stop()
    >>> print(watcher.get_alert_log())
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[WatchConfig] = None,
    ) -> None:

        if not isinstance(model, nn.Module):
            raise TypeError(
                f"[moewatch] MoEWatch expects a torch.nn.Module, "
                f"got {type(model).__name__}."
            )

        self.model  = model
        self.config = config if config is not None else WatchConfig()

        # Internal state — always starts detached
        self._hook_manager:   Optional[Any] = None   # HookManager (lazy)
        self._collector:      Optional[Any] = None   # StatCollector (lazy)
        self._attached:       bool          = False
        self._global_step:    int           = 0
        self._alert_log:      List[Alert]   = []
        self._start_time:     Optional[float] = None

        # Sub-module references resolved lazily to keep import fast
        self._HookManager        = None
        self._detect_routers     = None
        self._StatCollector      = None
        self._EntropyAnalyzer    = None
        self._CollapseDetector   = None
        self._entropy_analyzer:  Optional[Any] = None
        self._collapse_detector: Optional[Any] = None

        log.debug("[moewatch] MoEWatch created. Model: %s", type(model).__name__)

    # -----------------------------------------------------------------------
    # Lazy sub-module bootstrapping
    # -----------------------------------------------------------------------

    def _bootstrap(self) -> None:
        """Resolve all lazy imports and wire up the hook infrastructure."""

        from .hooks.manager    import HookManager
        from .hooks.detection  import detect_router_modules
        from .collector.stat_collector import StatCollector
        from .analyzer.entropy  import EntropyAnalyzer
        from .analyzer.collapse import CollapseDetector

        self._HookManager       = HookManager
        self._detect_routers    = detect_router_modules
        self._StatCollector     = StatCollector
        self._EntropyAnalyzer   = EntropyAnalyzer
        self._CollapseDetector  = CollapseDetector

    # -----------------------------------------------------------------------
    # Attach / detach lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> "MoEWatch":
        """Attach hooks and begin monitoring. Returns self for chaining.

        Call this before the first training step when using a custom loop.
        Idempotent — calling ``start()`` on an already-running watcher is
        a no-op with a warning.
        """
        if self._attached:
            warnings.warn(
                "[moewatch] MoEWatch is already running. "
                "Ignoring duplicate start() call.",
                UserWarning,
                stacklevel=2,
            )
            return self

        self._bootstrap()
        self._start_time = time.perf_counter()

        # -- Detect or accept router modules --------------------------------
        if self.config.router_modules:
            router_names = list(self.config.router_modules)
        else:
            router_names = self._detect_routers(self.model)

        if not router_names:
            raise RuntimeError(
                "[moewatch] Could not detect any router modules. "
                "Specify them via WatchConfig(router_modules=[...])."
            )

        # -- Wire up collector and hooks ------------------------------------
        self._collector = self._StatCollector(
            layer_names=router_names,
            config=self.config,
        )

        self._hook_manager = self._HookManager(
            model=self.model,
            router_module_names=router_names,
            collector=self._collector,
            config=self.config,
        )
        self._hook_manager.attach()
        self._entropy_analyzer  = self._EntropyAnalyzer(config=self.config)
        self._collapse_detector = self._CollapseDetector(config=self.config)
        self._attached = True

        self._print_banner(router_names)
        log.info(
            "[moewatch] Attached %d router hook(s). Monitoring started.",
            len(router_names),
        )
        return self

    def stop(self) -> "MoEWatch":
        """Detach all hooks and stop monitoring. Returns self for chaining.

        Safe to call multiple times. Always cleans up hooks — even if a prior
        exception occurred — so no hooks will be leaked in your model.
        """
        if not self._attached:
            return self

        if self._hook_manager is not None:
            try:
                self._hook_manager.detach()
            except Exception as exc:                      # pragma: no cover
                log.warning(
                    "[moewatch] Non-fatal error during hook detach: %s", exc
                )
            finally:
                self._hook_manager = None

        self._attached    = False
        self._collector   = None

        elapsed = (
            time.perf_counter() - self._start_time
            if self._start_time
            else 0.0
        )
        log.info(
            "[moewatch] Monitoring stopped after %.1f s. "
            "%d alert(s) emitted.",
            elapsed,
            len(self._alert_log),
        )
        return self

    def attach(self, trainer: Any) -> "MoEWatch":
        """Attach moewatch to a HuggingFace ``Trainer``.

        Injects a ``MoEWatchCallback`` into the trainer's callback list and
        calls ``start()`` to wire up the hooks immediately.

        Parameters
        ----------
        trainer : transformers.Trainer
            Any HuggingFace Trainer instance. moewatch never modifies the
            trainer's model, optimiser, or data pipeline.

        Returns
        -------
        MoEWatch
            Returns self for method chaining.

        Raises
        ------
        TypeError
            If *trainer* does not look like a HuggingFace Trainer (i.e. has
            no ``add_callback`` method).
        """
        if not hasattr(trainer, "add_callback"):
            raise TypeError(
                "[moewatch] attach() expects a HuggingFace Trainer with an "
                "'add_callback' method. For custom training loops, use "
                "watcher.start() and watcher.step(step_number) instead."
            )

        # Start hook infrastructure first
        self.start()

        # Register the callback that calls watcher.step() at each log step
        callback = MoEWatchCallback(watcher=self)
        trainer.add_callback(callback)

        log.info(
            "[moewatch] MoEWatchCallback registered with HuggingFace Trainer."
        )
        return self

    def detach(self) -> "MoEWatch":
        """Alias for ``stop()``. Consistent with PyTorch hook terminology."""
        return self.stop()

    # -----------------------------------------------------------------------
    # Step-level diagnostic tick
    # -----------------------------------------------------------------------

    def step(self, global_step: int) -> List[Alert]:
        """Run a diagnostic check at *global_step*.

        This is called automatically by ``MoEWatchCallback`` when using the
        HuggingFace Trainer integration. Call it manually from a custom loop.

        Parameters
        ----------
        global_step : int
            Current training step. Used for alert messages and the alert log.

        Returns
        -------
        list of Alert
            Alerts fired at this step (empty list if nothing is wrong).
        """
        if not self._attached or self._collector is None:
            return []

        self._global_step = global_step

        # Only run analysis every log_every steps
        if global_step % self.config.log_every != 0:
            return []

        step_alerts: List[Alert] = []

        try:
            layer_stats = self._collector.get_all_stats()
        except Exception as exc:                          # pragma: no cover
            log.warning(
                "[moewatch] Failed to retrieve stats at step %d: %s",
                global_step,
                exc,
            )
            return []

        if not layer_stats:
            # Hooks attached but no data yet — model hasn't run a forward pass
            return []

        # -- Entropy analysis -----------------------------------------------
        entropy_report = self._entropy_analyzer.analyze(layer_stats)

        for layer_name, result in entropy_report.results.items():
            if result.alert_level == AlertLevel.ERROR:
                alert = Alert(
                    step=global_step,
                    level=AlertLevel.ERROR,
                    layer_name=layer_name,
                    message=(
                        f"Routing entropy CRITICAL: {result.entropy_norm:.1%} "
                        f"of max entropy. Router is severely collapsed."
                    ),
                    metric=result.entropy_norm,
                    suggestion=_SUGGESTIONS["low_entropy"],
                )
                step_alerts.append(alert)

            elif result.alert_level == AlertLevel.WARN:
                alert = Alert(
                    step=global_step,
                    level=AlertLevel.WARN,
                    layer_name=layer_name,
                    message=(
                        f"Routing entropy LOW: {result.entropy_norm:.1%} "
                        f"of max entropy."
                    ),
                    metric=result.entropy_norm,
                )
                step_alerts.append(alert)

        # -- Collapse analysis ----------------------------------------------
        collapse_results = self._collapse_detector.detect(layer_stats)

        for layer_name, report in collapse_results.items():
            for status in report.experts:
                if status.is_dead:
                    alert = Alert(
                        step=global_step,
                        level=AlertLevel.ERROR,
                        layer_name=layer_name,
                        message=(
                            f"Expert {status.expert_idx} DEAD — "
                            f"utilisation: {status.utilization:.3%}"
                        ),
                        metric=status.utilization,
                        suggestion=_SUGGESTIONS["dead_expert"],
                    )
                    step_alerts.append(alert)

                elif status.is_cold:
                    alert = Alert(
                        step=global_step,
                        level=AlertLevel.WARN,
                        layer_name=layer_name,
                        message=(
                            f"Expert {status.expert_idx} COLD — "
                            f"utilisation: {status.utilization:.3%}"
                        ),
                        metric=status.utilization,
                    )
                    step_alerts.append(alert)

        # -- Load imbalance -------------------------------------------------
        for layer_name, stats in layer_stats.items():
            imbalance = stats.load_imbalance_score
            if imbalance != imbalance:  # NaN check
                continue
            if imbalance > self.config.load_imbalance_error:
                alert = Alert(
                    step=global_step,
                    level=AlertLevel.ERROR,
                    layer_name=layer_name,
                    message=(
                        f"Load imbalance CRITICAL: max/mean = {imbalance:.1f}×"
                    ),
                    metric=imbalance,
                    suggestion=_SUGGESTIONS["load_imbalance"],
                )
                step_alerts.append(alert)
            elif imbalance > self.config.load_imbalance_warn:
                alert = Alert(
                    step=global_step,
                    level=AlertLevel.WARN,
                    layer_name=layer_name,
                    message=(
                        f"Load imbalance: max/mean = {imbalance:.1f}×"
                    ),
                    metric=imbalance,
                )
                step_alerts.append(alert)

        # -- Healthy heartbeat (emit when nothing is wrong) -----------------
        if not step_alerts:
            heartbeat = Alert(
                step=global_step,
                level=AlertLevel.INFO,
                layer_name="*",
                message="All experts active. Routing health nominal.",
            )
            step_alerts.append(heartbeat)

        # -- Store and emit -------------------------------------------------
        self._alert_log.extend(step_alerts)
        self._emit_alerts(step_alerts)

        return step_alerts

    # -----------------------------------------------------------------------
    # Alert emission
    # -----------------------------------------------------------------------

    def _emit_alerts(self, alerts: List[Alert]) -> None:
        """Format and write *alerts* to the configured output channel."""

        if self.config.output == OutputMode.SILENT:
            return

        for alert in alerts:
            if self.config.output == OutputMode.JSON:
                print(alert.to_json_line(), flush=True)

            else:  # CONSOLE
                icon   = _LEVEL_ICON[alert.level]
                colour = _LEVEL_COLOUR[alert.level]
                no_col = self.config.no_color

                step_tag = _colour(
                    f"[Step {alert.step:>6}]",
                    _ANSI_CYAN + _ANSI_BOLD,
                    no_color=no_col,
                )
                level_tag = _colour(
                    f"{alert.level.value:<5}",
                    colour + _ANSI_BOLD,
                    no_color=no_col,
                )
                msg = _colour(alert.message, colour, no_color=no_col)

                print(
                    f"  {icon}  {step_tag}  {level_tag}  {msg}",
                    file=sys.stdout,
                    flush=True,
                )

                if alert.suggestion and alert.level == AlertLevel.ERROR:
                    suggestion_prefix = _colour("         💡 Suggestion:", _ANSI_BOLD, no_color=no_col)
                    print(
                        f"{suggestion_prefix} {alert.suggestion}",
                        file=sys.stdout,
                        flush=True,
                    )

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def get_alert_log(self) -> List[Alert]:
        """Return all alerts emitted since monitoring started.

        Returns a copy — mutating the returned list has no effect on the
        internal log.
        """
        return list(self._alert_log)

    def get_alert_log_json(self) -> str:
        """Return all alerts as a JSON array string."""
        return json.dumps([a.to_dict() for a in self._alert_log], indent=2)

    @property
    def is_attached(self) -> bool:
        """True if hooks are currently active."""
        return self._attached

    @property
    def global_step(self) -> int:
        """Last training step seen by the watcher."""
        return self._global_step

    def summary(self) -> str:
        """Return a brief text summary of monitoring results."""
        errors = [a for a in self._alert_log if a.level == AlertLevel.ERROR]
        warns  = [a for a in self._alert_log if a.level == AlertLevel.WARN]
        return (
            f"moewatch summary — {len(self._alert_log)} total alert(s): "
            f"{len(errors)} ERROR, {len(warns)} WARN, "
            f"{len(self._alert_log) - len(errors) - len(warns)} INFO"
        )

    def __repr__(self) -> str:
        status = "attached" if self._attached else "detached"
        return (
            f"MoEWatch("
            f"model={type(self.model).__name__}, "
            f"status={status}, "
            f"step={self._global_step}, "
            f"alerts={len(self._alert_log)}"
            f")"
        )

    # -----------------------------------------------------------------------
    # Context manager interface (optional ergonomic alternative)
    # -----------------------------------------------------------------------

    def __enter__(self) -> "MoEWatch":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    # -----------------------------------------------------------------------
    # Pretty header
    # -----------------------------------------------------------------------

    def _print_banner(self, router_names: List[str]) -> None:
        if self.config.output == OutputMode.SILENT:
            return

        _LINE = "─" * 66
        nc    = self.config.no_color

        print()
        print(f"  ╔{_LINE}╗")
        print(f"  ║{_colour('moewatch · Live Training Monitor', _ANSI_BOLD, no_color=nc):^66}║")
        print(f"  ╠{_LINE}╣")
        print(f"  ║  {'Router modules attached':.<40} {len(router_names):>8}        ║")
        print(f"  ║  {'Log every N steps':.<40} {self.config.log_every:>8}        ║")
        print(f"  ║  {'Sample every N steps':.<40} {self.config.sample_every:>8}        ║")
        print(f"  ║  {'Dead threshold':.<40} {self.config.dead_threshold:>7.3%}        ║")
        print(f"  ║  {'Alert output mode':.<40} {self.config.output.value:>8}        ║")
        print(f"  ╠{_LINE}╣")
        for name in router_names[:5]:  # show first 5
            truncated = name[-54:] if len(name) > 54 else name
            print(f"  ║  ↳ {truncated:<62}║")
        if len(router_names) > 5:
            remaining = len(router_names) - 5
            print(f"  ║  ↳ ... and {remaining} more layer(s){' ' * (52 - len(str(remaining)))}║")
        print(f"  ╚{_LINE}╝")
        print()


# ---------------------------------------------------------------------------
# HuggingFace TrainerCallback integration
# ---------------------------------------------------------------------------

try:
    from transformers import TrainerCallback as _TrainerCallbackBase
    _HF_AVAILABLE = True
except ImportError:
    _TrainerCallbackBase = object
    _HF_AVAILABLE = False

class MoEWatchCallback(_TrainerCallbackBase):
    """HuggingFace ``TrainerCallback`` that ticks ``MoEWatch.step()`` during training.

    This class is intentionally thin. All diagnostic logic lives in
    ``MoEWatch.step()``; the callback is only responsible for forwarding
    the current training step from the HuggingFace training loop.

    You do not normally instantiate this directly — ``MoEWatch.attach(trainer)``
    creates and registers it for you.

    Parameters
    ----------
    watcher : MoEWatch
        The parent watcher instance. Must already be started (hooks attached)
        before the callback fires.
    """

    def __init__(self, watcher: MoEWatch) -> None:
        self.watcher = watcher
        if _HF_AVAILABLE:
            super().__init__()

    # Called by HuggingFace Trainer at every logging step
    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Tick the watcher at each HuggingFace logging event."""
        global_step = getattr(state, "global_step", 0)
        self.watcher.step(global_step)

    # Called at training end — ensure hooks are cleaned up
    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Detach hooks when HuggingFace Trainer completes training."""
        if self.watcher.is_attached:
            self.watcher.stop()

    def __repr__(self) -> str:
        return f"MoEWatchCallback(watcher={self.watcher!r})"
