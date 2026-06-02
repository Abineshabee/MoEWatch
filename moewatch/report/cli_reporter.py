# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  report/cli_reporter.py — Coloured ASCII terminal renderer for AuditReport
#
#  CLIReporter is the sole owner of all terminal output logic. It reads an
#  AuditReport (pure data) and renders it to stdout as a structured,
#  colour-coded diagnostic report. It has zero side effects on the report.
#
#  Design principles
#  -----------------
#    - Strict separation of data (AuditReport) from presentation (CLIReporter).
#      AuditReport never imports CLIReporter; the dependency is one-way.
#    - NO_COLOR compliance: all ANSI codes are suppressed when the NO_COLOR
#      environment variable is set or when WatchConfig.no_color is True.
#    - Width-safe: all output lines are capped at TERMINAL_WIDTH (default 80)
#      to avoid wrapping on narrow terminals. Layer names are truncated, never
#      wrapped, so the histogram columns stay aligned.
#    - Zero-crash guarantee: every render method catches its own exceptions
#      and falls back to a minimal plain-text output rather than letting a
#      display bug propagate to the researcher's training script.
#    - Stdout-only: CLIReporter always writes to sys.stdout. Redirect via
#      shell piping or WatchConfig(output=OutputMode.SILENT) if needed.
#
#  Sections rendered by print_summary()
#  -------------------------------------
#    §A  Report header (metadata, health banner)
#    §B  Entropy section (per-layer entropy bar + alert level)
#    §C  Expert utilization histogram (per-layer, per-expert ASCII bar)
#    §D  Dead / cold expert table
#    §E  Recommendations
#    §F  Footer (elapsed time, config snapshot link)
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#  Version: 0.1.0
#
# =============================================================================

from __future__ import annotations

import math
import os
import sys
import textwrap
from typing import Optional

from ..config import AlertLevel, WatchConfig
from .audit_report import (
    AuditReport,
    OverallHealth,
    UtilizationSummary,
    _shorten_layer_name,
)
from ..analyzer.entropy import EntropyResult, TrendDirection
from ..analyzer.collapse import ExpertState

# =============================================================================
# Section 1 — ANSI colour palette
# =============================================================================
#
# All colour constants are defined here. Nothing outside this section should
# reference raw ANSI escape sequences — use the _c() / _bold() helpers instead.
# =============================================================================

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

# Foreground colours
_FG_RED = "\033[31m"
_FG_GREEN = "\033[32m"
_FG_YELLOW = "\033[33m"
_FG_BLUE = "\033[34m"
_FG_MAGENTA = "\033[35m"
_FG_CYAN = "\033[36m"
_FG_WHITE = "\033[37m"

# Bright foreground colours (bolder on most terminals)
_FG_BRED = "\033[91m"
_FG_BGREEN = "\033[92m"
_FG_BYELLOW = "\033[93m"
_FG_BCYAN = "\033[96m"
_FG_BWHITE = "\033[97m"

# Background colours (used for the health banner only)
_BG_RED = "\033[41m"
_BG_GREEN = "\033[42m"
_BG_YELLOW = "\033[43m"
_BG_BLUE = "\033[44m"

# Mappings: AlertLevel → foreground colour
_ALERT_FG = {
    AlertLevel.INFO: _FG_BGREEN,
    AlertLevel.WARN: _FG_BYELLOW,
    AlertLevel.ERROR: _FG_BRED,
}

# Mappings: OverallHealth → (background, foreground)
_HEALTH_BANNER_COLOUR = {
    OverallHealth.HEALTHY: (_BG_GREEN, _FG_WHITE),
    OverallHealth.DEGRADING: (_BG_YELLOW, _FG_WHITE),
    OverallHealth.CRITICAL: (_BG_RED, _FG_BWHITE),
    OverallHealth.UNKNOWN: (_BG_BLUE, _FG_BWHITE),
}

# Icons for alert levels and expert states
_ICON_OK = "✅"
_ICON_WARN = "⚠ "
_ICON_ERROR = "❌"
_ICON_COLD = "🥶"
_ICON_TIP = "💡"
_ICON_CHART = "📊"
_ICON_ROUTER = "🔀"

_TREND_SYMBOL = {
    TrendDirection.STABLE: "→",
    TrendDirection.IMPROVING: "↑",
    TrendDirection.DECLINING: "↓",
    TrendDirection.UNKNOWN: "·",
}


# =============================================================================
# Section 2 — Histogram characters
# =============================================================================

# Block elements for the ASCII bar: 1/8 increments
_BLOCKS = " ▏▎▍▌▋▊▉█"

# Number of bar fill characters for a "full" bar
_BAR_MAX_WIDTH = 20


# =============================================================================
# Section 3 — Layout constants
# =============================================================================

TERMINAL_WIDTH: int = 80
_INNER_WIDTH: int = TERMINAL_WIDTH - 4  # inside the box (2 × "  ")
_LINE_HEAVY: str = "═" * TERMINAL_WIDTH
_LINE_LIGHT: str = "─" * TERMINAL_WIDTH
_LINE_MED: str = "━" * TERMINAL_WIDTH


# =============================================================================
# Section 4 — CLIReporter
# =============================================================================


class CLIReporter:
    """Renders a coloured ASCII diagnostic report from an :class:`AuditReport`.

    Parameters
    ----------
    config : WatchConfig
        Controls ``no_color`` flag. If ``config.no_color`` is True, or if
        the ``NO_COLOR`` environment variable is set, all ANSI escape codes
        are stripped from the output.

    Examples
    --------
    >>> reporter = CLIReporter(config=WatchConfig())
    >>> reporter.print_summary(report)   # writes to stdout

    >>> reporter = CLIReporter(config=WatchConfig(no_color=True))
    >>> reporter.print_summary(report)   # same, without colour
    """

    def __init__(self, config: WatchConfig) -> None:
        self.config = config
        # Respect both the WatchConfig flag and the NO_COLOR env standard
        self.no_color: bool = config.no_color or ("NO_COLOR" in os.environ)

    # =========================================================================
    # §4.1  Top-level render method
    # =========================================================================

    def print_summary(self, report: AuditReport) -> None:
        """Render the full diagnostic report to stdout.

        Renders all six sections (header, entropy, utilization, dead experts,
        recommendations, footer) in sequence. Any rendering error in a section
        is caught and a fallback message is printed — the remaining sections
        continue unaffected.

        Parameters
        ----------
        report : AuditReport
            The report to render.  Must be a fully-constructed AuditReport.
        """
        out = sys.stdout

        try:
            self._render_header(report, out)
        except Exception as exc:  # pragma: no cover
            print(f"  [moewatch] Header render error: {exc}", file=out)

        try:
            self._render_entropy_section(report, out)
        except Exception as exc:  # pragma: no cover
            print(f"  [moewatch] Entropy section error: {exc}", file=out)

        try:
            self._render_utilization_section(report, out)
        except Exception as exc:  # pragma: no cover
            print(f"  [moewatch] Utilization section error: {exc}", file=out)

        try:
            self._render_dead_experts_section(report, out)
        except Exception as exc:  # pragma: no cover
            print(f"  [moewatch] Dead experts section error: {exc}", file=out)

        try:
            self._render_recommendations_section(report, out)
        except Exception as exc:  # pragma: no cover
            print(f"  [moewatch] Recommendations section error: {exc}", file=out)

        try:
            self._render_footer(report, out)
        except Exception as exc:  # pragma: no cover
            print(f"  [moewatch] Footer render error: {exc}", file=out)

        out.flush()

    # =========================================================================
    # §4.2  Section A — Header
    # =========================================================================

    def _render_header(self, report: AuditReport, out) -> None:
        """Render the report header: logo, health banner, run metadata."""

        self._println(self._c(_LINE_HEAVY, _BOLD), out)

        # -- moewatch wordmark ----------------------------------------------
        wordmark = "  moewatch  ·  MoE Diagnostic Audit Report  ·  v0.1.0"
        self._println(self._c(wordmark, _BOLD + _FG_BCYAN), out)

        self._println(self._c(_LINE_LIGHT, _DIM), out)

        # -- Health banner --------------------------------------------------
        health = report.overall_health
        bg, fg = _HEALTH_BANNER_COLOUR.get(health, (_BG_BLUE, _FG_BWHITE))
        icon = {
            OverallHealth.HEALTHY: _ICON_OK,
            OverallHealth.DEGRADING: _ICON_WARN,
            OverallHealth.CRITICAL: _ICON_ERROR,
            OverallHealth.UNKNOWN: "❓",
        }.get(health, "?")

        banner_text = f"  {icon}  Overall Health: {health.value:<12s}"
        padded = banner_text + " " * max(0, TERMINAL_WIDTH - len(banner_text))
        if not self.no_color:
            print(f"{bg}{fg}{_BOLD}{padded}{_RESET}", file=out)
        else:
            print(padded, file=out)

        self._println(self._c(_LINE_LIGHT, _DIM), out)

        # -- Run metadata table ----------------------------------------------
        meta_rows = [
            ("Router layers", str(report.n_layers)),
            ("Experts monitored", str(report.n_experts_total)),
            ("Completed steps", str(report.completed_steps)),
            ("Elapsed", f"{report.elapsed_seconds:.2f} s"),
            ("Device", report.device),
            ("Dead experts", self._coloured_count(report.n_dead, AlertLevel.ERROR)),
            ("Cold experts", self._coloured_count(report.n_cold, AlertLevel.WARN)),
            (
                "Healthy experts",
                self._coloured_count(report.n_healthy, AlertLevel.INFO),
            ),
        ]
        for label, value in meta_rows:
            label_col = self._c(f"  {label:<28s}", _DIM)
            print(f"{label_col}{value}", file=out)

        self._println(self._c(_LINE_HEAVY, _BOLD), out)
        print("", file=out)

    # =========================================================================
    # §4.3  Section B — Entropy
    # =========================================================================

    def _render_entropy_section(self, report: AuditReport, out) -> None:
        """Render per-layer entropy bars with alert level and trend indicator."""

        entropy_results = report.routing_entropy()
        if not entropy_results:
            return

        # -- Section header -------------------------------------------------
        self._section_header(f"  {_ICON_CHART}  Routing Entropy", out)
        self._println(
            self._c(
                f"  {'Layer':<44s}  {'H (bits)':>8s}  {'/ max':>6s}  {'Alert':<7s}  Trend",
                _BOLD,
            ),
            out,
        )
        self._println(self._c("  " + "─" * 76, _DIM), out)

        for layer_name in report.layer_names():
            result = entropy_results.get(layer_name)
            if result is None:
                continue
            self._render_entropy_row(layer_name, result, out)

        # -- Entropy aggregate summary ------------------------------------
        er = report._entropy_results
        summary_parts = [
            f"  global: {self._alert_badge(er.global_alert)}",
            f"  warn={er.n_warn}",
            f"  error={er.n_error}",
            f"  declining={er.n_declining}",
        ]
        self._println("", out)
        self._println(self._c("  " + "  ".join(summary_parts), _DIM), out)
        self._println("", out)

    def _render_entropy_row(
        self,
        layer_name: str,
        result: EntropyResult,
        out,
    ) -> None:
        """Render a single layer's entropy as one formatted table row."""

        short_name = _shorten_layer_name(layer_name, max_len=44)

        if result.is_empty:
            dim_name = self._c(f"  {short_name:<44s}", _DIM)
            print(f"{dim_name}  {'N/A':>8s}  {'---':>6s}  {'NO DATA':<7s}", file=out)
            return

        alert = result.alert_level
        colour = _ALERT_FG.get(alert, _FG_WHITE)
        icon = (
            _ICON_OK
            if alert == AlertLevel.INFO
            else (_ICON_WARN if alert == AlertLevel.WARN else _ICON_ERROR)
        )

        trend_sym = _TREND_SYMBOL.get(result.trend, "·")
        trend_col = (
            _FG_BGREEN
            if result.trend == TrendDirection.IMPROVING
            else _FG_BRED if result.trend == TrendDirection.DECLINING else _FG_WHITE
        )
        trend_str = self._c(f"{trend_sym} {result.trend:<9s}", trend_col)

        name_str = self._c(f"  {short_name:<44s}", colour)
        h_str = f"{result.entropy_bits:>8.3f}"
        norm_str = self._c(f"{result.entropy_norm*100:>5.1f}%", colour)
        alert_badge = self._alert_badge(alert)

        print(
            f"{name_str}  {h_str}  {norm_str}  {icon} {alert_badge:<6s}  {trend_str}",
            file=out,
        )

    # =========================================================================
    # §4.4  Section C — Utilization histogram
    # =========================================================================

    def _render_utilization_section(self, report: AuditReport, out) -> None:
        """Render per-layer, per-expert ASCII utilization histograms."""

        util_dict = report.utilization()
        if not util_dict:
            return

        self._section_header(f"  {_ICON_ROUTER}  Expert Utilization", out)

        collapse_results = report._collapse_results
        layer_names = report.layer_names()

        for layer_name in layer_names:
            summary = util_dict.get(layer_name)
            if summary is None:
                continue

            collapse = collapse_results.get(layer_name)
            self._render_layer_histogram(layer_name, summary, collapse, out)

    def _render_layer_histogram(
        self,
        layer_name: str,
        summary: UtilizationSummary,
        collapse,  # Optional[LayerCollapseReport]
        out,
    ) -> None:
        """Render a single layer's per-expert utilization histogram."""

        short_name = _shorten_layer_name(layer_name, max_len=60)

        # -- Layer sub-header -----------------------------------------------
        lim = summary.load_imbalance_score
        lim_str = f"{lim:.2f}×" if not math.isnan(lim) else "N/A"
        lim_col = (
            _FG_BRED
            if not math.isnan(lim) and lim > self.config.load_imbalance_error
            else (
                _FG_BYELLOW
                if not math.isnan(lim) and lim > self.config.load_imbalance_warn
                else _FG_BGREEN
            )
        )
        header = (
            f"  ┌─ {self._c(short_name, _BOLD)}  "
            f"imbalance={self._c(lim_str, lim_col)}  "
            f"events={summary.event_count}"
        )
        self._println(header, out)

        n = summary.n_experts

        # -- Decide how many experts to show --------------------------------
        # Always show dead/cold experts even if we abbreviate the rest.
        problematic_idxs: set = set()
        if collapse and not collapse.is_empty:
            for expert in collapse.experts:
                if expert.is_problematic:
                    problematic_idxs.add(expert.expert_idx)

        # Show all experts when n ≤ 32; otherwise abbreviate healthy ones.
        show_all = n <= 32

        shown_count = 0
        abbreviated = False

        for idx in range(n):
            util_frac = (
                summary.utilization[idx] if idx < len(summary.utilization) else 0.0
            )
            count = summary.token_counts[idx] if idx < len(summary.token_counts) else 0

            # Expert state for colour
            state: Optional[ExpertState] = None
            if collapse and not collapse.is_empty:
                expert_status = collapse.expert(idx)
                if expert_status is not None:
                    state = expert_status.state

            is_prob = idx in problematic_idxs

            if not show_all and not is_prob:
                # Abbreviated: only show every 4th healthy expert
                if idx % 4 != 0:
                    abbreviated = True
                    continue

            self._render_expert_bar(idx, util_frac, count, state, out)
            shown_count += 1

        if abbreviated:
            self._println(
                self._c(f"  │  ... ({n - shown_count} healthy experts hidden)", _DIM),
                out,
            )

        # -- Layer footer ---------------------------------------------------
        self._println(
            self._c(
                f"  └─ Layer total: {summary.total_tokens:,} tokens  |  "
                f"max={summary.max_util*100:.2f}%  "
                f"min={summary.min_util*100:.3f}%  "
                f"mean={summary.mean_util*100:.2f}%",
                _DIM,
            ),
            out,
        )
        print("", file=out)

    def _render_expert_bar(
        self,
        idx: int,
        util_frac: float,
        count: int,
        state: Optional[ExpertState],
        out,
    ) -> None:
        """Render a single expert's utilization bar line.

        Format:
          │  Expert  7  │█████████░░░░░░░░░░░│  12.34%  (1234 tokens)  [OK]
        """
        # -- Determine colour and icon based on expert state ----------------
        if state == ExpertState.DEAD:
            bar_colour = _FG_BRED
            state_tag = self._c("[DEAD]", _FG_BRED + _BOLD)
        elif state == ExpertState.COLD:
            bar_colour = _FG_BYELLOW
            state_tag = self._c("[COLD]", _FG_BYELLOW)
        else:
            # Colour by utilization quartile
            if util_frac >= 0.15:
                bar_colour = _FG_BGREEN
            elif util_frac >= 0.05:
                bar_colour = _FG_GREEN
            elif util_frac >= 0.01:
                bar_colour = _FG_BYELLOW
            else:
                bar_colour = _FG_BRED
            state_tag = (
                self._c("[OK]  ", _FG_BGREEN)
                if util_frac >= 0.01
                else self._c("[LOW] ", _FG_BYELLOW)
            )

        # -- Build the ASCII bar --------------------------------------------
        bar = _build_bar(util_frac, width=_BAR_MAX_WIDTH)
        bar_str = self._c(bar, bar_colour)

        pct_str = f"{util_frac * 100:>6.3f}%"
        count_str = self._c(f"({count:>7,} tok)", _DIM)

        line = (
            f"  │  Expert {idx:>3d}  │{bar_str}│  "
            f"{pct_str}  {count_str}  {state_tag}"
        )
        print(line, file=out)

    # =========================================================================
    # §4.5  Section D — Dead / cold expert table
    # =========================================================================

    def _render_dead_experts_section(self, report: AuditReport, out) -> None:
        """Render a compact table of all dead and cold experts."""

        dead_list = report.dead_experts(include_cold=True)
        if not dead_list:
            return

        dead_only = [e for e in dead_list if e.is_dead]
        cold_only = [e for e in dead_list if e.is_cold]

        self._section_header(
            f"  {_ICON_ERROR}  Expert Collapse Summary  "
            f"({len(dead_only)} dead / {len(cold_only)} cold)",
            out,
        )
        self._println(
            self._c(
                f"  {'Layer':<42s}  {'Exp':>4s}  {'State':<8s}  {'Util %':>7s}  {'Cold steps':>10s}",
                _BOLD,
            ),
            out,
        )
        self._println(self._c("  " + "─" * 76, _DIM), out)

        for entry in dead_list:
            short = _shorten_layer_name(entry.layer_name, max_len=42)
            if entry.is_dead:
                col = _FG_BRED
                icon = _ICON_ERROR
            else:
                col = _FG_BYELLOW
                icon = _ICON_COLD

            state_str = self._c(f"{entry.state.value:<8s}", col + _BOLD)
            util_str = self._c(f"{entry.utilization_pct:>7.4f}", col)
            cold_str = self._c(f"{entry.consecutive_cold_steps:>10d}", _DIM)
            name_str = self._c(f"  {short:<42s}", col)

            print(
                f"{icon}{name_str}  {entry.expert_idx:>4d}  "
                f"{state_str}  {util_str}  {cold_str}",
                file=out,
            )

        self._println("", out)

    # =========================================================================
    # §4.6  Section E — Recommendations
    # =========================================================================

    def _render_recommendations_section(self, report: AuditReport, out) -> None:
        """Render prioritised actionable recommendations."""

        recs = report.recommendations()
        if not recs:
            return

        # Only show this section if there are actual problems to recommend on
        is_healthy = (
            report.overall_health == OverallHealth.HEALTHY
            and len(recs) == 1
            and "[INFO]" in recs[0]
        )

        self._section_header(f"  {_ICON_TIP}  Recommendations", out)

        if is_healthy:
            self._println(
                self._c(
                    f"  {_ICON_OK}  All experts active and entropy healthy. No action required.",
                    _FG_BGREEN,
                ),
                out,
            )
            self._println("", out)
            return

        for rec in recs:
            # Parse severity prefix from the recommendation string
            if rec.startswith("[CRITICAL]"):
                colour = _FG_BRED
                icon = _ICON_ERROR
                text = rec[len("[CRITICAL]") :].lstrip()
            elif rec.startswith("[WARN]"):
                colour = _FG_BYELLOW
                icon = _ICON_WARN
                text = rec[len("[WARN]") :].lstrip()
            else:
                colour = _FG_BGREEN
                icon = _ICON_OK
                text = (
                    rec[len("[INFO]") :].lstrip() if rec.startswith("[INFO]") else rec
                )

            # Print the first sentence in colour, continuation lines dimmed
            sentences = text.split(". ", 1)
            first_line = self._c(f"  {icon}  {sentences[0]}.", colour + _BOLD)
            self._println(first_line, out)

            if len(sentences) > 1:
                continuation = sentences[1].strip()
                wrapped = textwrap.fill(
                    continuation,
                    width=74,
                    initial_indent="       ",
                    subsequent_indent="       ",
                )
                self._println(self._c(wrapped, _DIM), out)

            self._println("", out)

    # =========================================================================
    # §4.7  Section F — Footer
    # =========================================================================

    def _render_footer(self, report: AuditReport, out) -> None:
        """Render the report footer with elapsed time and config summary."""

        self._println(self._c(_LINE_HEAVY, _BOLD), out)

        # -- Timing line ----------------------------------------------------
        timing = (
            f"  Audit completed in {report.elapsed_seconds:.2f} s  ·  "
            f"{report.completed_steps} forward passes  ·  "
            f"device: {report.device}"
        )
        self._println(self._c(timing, _DIM), out)

        # -- Config digest --------------------------------------------------
        cfg = report.config
        config_digest = (
            f"  Config: dead_threshold={cfg.dead_threshold:.4f}  "
            f"entropy_warn={cfg.entropy_warn:.2f}  "
            f"sample_every={cfg.sample_every}  "
            f"window_steps={cfg.window_steps}"
        )
        self._println(self._c(config_digest, _DIM), out)

        # -- Docs / GH link -------------------------------------------------
        link = self._c(
            "  Docs: https://github.com/Abineshabee/moewatch",
            _FG_CYAN + _DIM,
        )
        self._println(link, out)

        self._println(self._c(_LINE_HEAVY, _BOLD), out)
        print("", file=out)

    # =========================================================================
    # §4.8  Section helpers
    # =========================================================================

    def _section_header(self, title: str, out) -> None:
        """Print a section header line with a light separator below."""
        self._println(self._c(_LINE_MED, _DIM), out)
        self._println(self._c(title, _BOLD + _FG_BCYAN), out)
        self._println(self._c(_LINE_LIGHT, _DIM), out)

    def _println(self, text: str, out) -> None:
        """Print *text* followed by a newline, without extra buffering."""
        print(text, file=out)

    # =========================================================================
    # §4.9  Colour helpers
    # =========================================================================

    def _c(self, text: str, code: str) -> str:
        """Apply ANSI *code* to *text*, respecting the no-colour flag.

        Returns the text unchanged when colour is disabled.
        """
        if self.no_color or not code:
            return text
        return f"{code}{text}{_RESET}"

    def _alert_badge(self, level: AlertLevel) -> str:
        """Return a coloured short string badge for *level*."""
        label = level.value
        colour = _ALERT_FG.get(level, _FG_WHITE)
        return self._c(label, colour + _BOLD)

    def _coloured_count(self, count: int, level: AlertLevel) -> str:
        """Return *count* as a coloured string.

        Zero counts are always shown in dim white (not alarming).
        Non-zero counts are coloured by alert level.
        """
        if count == 0:
            return self._c("0", _DIM)
        colour = _ALERT_FG.get(level, _FG_WHITE)
        return self._c(str(count), colour + _BOLD)


# =============================================================================
# Section 5 — ASCII bar builder
# =============================================================================


def _build_bar(fraction: float, width: int = _BAR_MAX_WIDTH) -> str:
    """Build a fixed-width Unicode block-element bar for *fraction* ∈ [0, 1].

    Uses 1/8-block elements for sub-character precision. A fraction of 0.0
    returns an empty bar; 1.0 returns a full bar.

    Parameters
    ----------
    fraction : float
        Value in [0.0, 1.0]. Values outside this range are clamped.
    width : int
        Total bar width in characters. Default 20.

    Returns
    -------
    str
        Bar string of exactly *width* characters, padded with spaces.

    Examples
    --------
    >>> _build_bar(0.5, width=10)
    '█████     '
    >>> _build_bar(0.0, width=10)
    '          '
    >>> _build_bar(1.0, width=10)
    '██████████'
    """
    clamped = max(0.0, min(1.0, fraction))
    total_eighths = round(clamped * width * 8)

    full_blocks = total_eighths // 8
    remainder = total_eighths % 8

    bar = "█" * full_blocks

    if remainder > 0 and full_blocks < width:
        bar += _BLOCKS[remainder]
        full_blocks += 1

    # Pad to exactly *width* characters
    bar = bar.ljust(width, " ")

    return bar
