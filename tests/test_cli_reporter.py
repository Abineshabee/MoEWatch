# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  tests/test_cli_reporter.py — Unit tests for CLIReporter
#
#  COVERAGE TARGETS
#  ----------------
#    _build_bar()
#      ├── Empty bar rendering
#      ├── Full bar rendering
#      ├── Negative-value clamping
#      ├── Over-1.0 clamping
#      └── Partial bar rendering
#
#    ANSI helpers
#      ├── _c() with no_color=True
#      ├── _c() with colours enabled
#      ├── _alert_badge() for INFO/WARN/ERROR
#      ├── _coloured_count() for zero values
#      └── _coloured_count() for non-zero values
#
#    Header & footer rendering
#      ├── Header generation
#      ├── Health summary display
#      ├── Metadata display
#      ├── Footer generation
#      └── Configuration summary display
#
#    Entropy rendering
#      ├── Empty entropy section
#      ├── Entropy section rendering
#      ├── Empty-layer handling
#      ├── INFO/WARN alert display
#      └── Trend rendering
#
#    Utilization rendering
#      ├── Expert DEAD state
#      ├── Expert COLD state
#      ├── Expert LOW state
#      ├── Expert OK state
#      └── Large-layer histogram abbreviation
#
#    Dead-expert reporting
#      ├── Empty dead-expert list
#      ├── Dead expert rendering
#      └── Cold expert rendering
#
#    Recommendation rendering
#      ├── Empty recommendations
#      ├── Healthy recommendations
#      ├── WARN recommendations
#      ├── CRITICAL recommendations
#      └── Multi-sentence recommendations
#
#    print_summary()
#      ├── Successful rendering path
#      └── Renderer exception recovery paths
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#
# =============================================================================

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from moewatch import WatchConfig
from moewatch.config import AlertLevel
from moewatch.report.cli_reporter import CLIReporter, _build_bar
from moewatch.report.audit_report import OverallHealth
from moewatch.analyzer.entropy import TrendDirection
from moewatch.analyzer.collapse import ExpertState


class TestCLIReporterCoverage:

    # =========================================================================
    # _build_bar
    # =========================================================================

    def test_build_bar_zero(self):
        bar = _build_bar(0.0, width=10)
        assert len(bar) == 10
        assert bar.strip() == ""

    def test_build_bar_full(self):
        assert _build_bar(1.0, width=10) == "██████████"

    def test_build_bar_negative_clamped(self):
        bar = _build_bar(-5.0, width=10)
        assert bar.strip() == ""

    def test_build_bar_above_one_clamped(self):
        assert _build_bar(5.0, width=10) == "██████████"

    def test_build_bar_partial(self):
        bar = _build_bar(0.37, width=10)
        assert len(bar) == 10
        assert any(ch != " " for ch in bar)

    # =========================================================================
    # Reporter helpers
    # =========================================================================

    @pytest.fixture()
    def reporter(self):
        return CLIReporter(WatchConfig(no_color=True))

    @pytest.fixture()
    def colored_reporter(self):
        return CLIReporter(WatchConfig(no_color=False))

    @pytest.fixture()
    def fake_report(self):

        report = SimpleNamespace()

        report.overall_health = OverallHealth.HEALTHY

        report.n_layers = 2
        report.n_experts_total = 16
        report.completed_steps = 100
        report.elapsed_seconds = 1.5
        report.device = "cpu"

        report.n_dead = 0
        report.n_cold = 1
        report.n_healthy = 15

        report.config = SimpleNamespace(
            dead_threshold=0.001,
            entropy_warn=0.75,
            sample_every=1,
            window_steps=50,
        )

        report.layer_names = lambda: ["router.layer0"]

        report.routing_entropy = lambda: {}

        report.utilization = lambda: {}

        report.dead_experts = lambda include_cold=True: []

        report.recommendations = lambda: [
            "[INFO] Everything looks healthy."
        ]

        report._entropy_results = SimpleNamespace(
            global_alert=AlertLevel.INFO,
            n_warn=0,
            n_error=0,
            n_declining=0,
        )

        report._collapse_results = {}

        return report

    # =========================================================================
    # ANSI helpers
    # =========================================================================

    def test_color_disabled(self, reporter):
        assert reporter._c("hello", "\033[31m") == "hello"

    def test_color_enabled(self, colored_reporter):
        result = colored_reporter._c("hello", "\033[31m")
        assert "\033[31m" in result
        assert "hello" in result

    @pytest.mark.parametrize(
        "level",
        [
            AlertLevel.INFO,
            AlertLevel.WARN,
            AlertLevel.ERROR,
        ],
    )
    def test_alert_badge(self, reporter, level):
        text = reporter._alert_badge(level)
        assert level.value in text

    @pytest.mark.parametrize(
        "level",
        [
            AlertLevel.INFO,
            AlertLevel.WARN,
            AlertLevel.ERROR,
        ],
    )
    def test_coloured_count_nonzero(self, reporter, level):
        assert "5" in reporter._coloured_count(5, level)

    def test_coloured_count_zero(self, reporter):
        assert reporter._coloured_count(0, AlertLevel.ERROR) == "0"

    # =========================================================================
    # Header / Footer
    # =========================================================================

    def test_render_header(self, reporter, fake_report):

        out = io.StringIO()

        reporter._render_header(fake_report, out)

        text = out.getvalue()

        assert "Overall Health" in text
        assert "Router layers" in text
        assert "Experts monitored" in text

    def test_render_footer(self, reporter, fake_report):

        out = io.StringIO()

        reporter._render_footer(fake_report, out)

        text = out.getvalue()

        assert "Audit completed" in text
        assert "Config:" in text
        assert "Docs:" in text

    # =========================================================================
    # Entropy
    # =========================================================================

    def test_entropy_section_empty(self, reporter, fake_report):

        out = io.StringIO()

        reporter._render_entropy_section(
            fake_report,
            out,
        )

        assert out.getvalue() == ""

    def test_entropy_row_no_data(self, reporter):

        result = SimpleNamespace(
            is_empty=True,
        )

        out = io.StringIO()

        reporter._render_entropy_row(
            "layer",
            result,
            out,
        )

        assert "NO DATA" in out.getvalue()

    @pytest.mark.parametrize(
        "trend",
        [
            TrendDirection.STABLE,
            TrendDirection.IMPROVING,
            TrendDirection.DECLINING,
            TrendDirection.UNKNOWN,
        ],
    )
    def test_entropy_row_all_trends(
        self,
        reporter,
        trend,
    ):

        result = SimpleNamespace(
            is_empty=False,
            alert_level=AlertLevel.WARN,
            trend=trend,
            entropy_bits=1.234,
            entropy_norm=0.55,
        )

        out = io.StringIO()

        reporter._render_entropy_row(
            "layer",
            result,
            out,
        )

        text = out.getvalue()

        assert "1.234" in text
        assert "WARN" in text

    def test_entropy_section_full(self, reporter, fake_report):

        result = SimpleNamespace(
            is_empty=False,
            alert_level=AlertLevel.INFO,
            trend=TrendDirection.STABLE,
            entropy_bits=2.0,
            entropy_norm=0.8,
        )

        fake_report.routing_entropy = lambda: {
            "router.layer0": result
        }

        out = io.StringIO()

        reporter._render_entropy_section(
            fake_report,
            out,
        )

        assert "Routing Entropy" in out.getvalue()

    # =========================================================================
    # Expert bars
    # =========================================================================

    def test_expert_bar_dead(self, reporter):

        out = io.StringIO()

        reporter._render_expert_bar(
            0,
            0.0,
            0,
            ExpertState.DEAD,
            out,
        )

        assert "[DEAD]" in out.getvalue()

    def test_expert_bar_cold(self, reporter):

        out = io.StringIO()

        reporter._render_expert_bar(
            0,
            0.001,
            1,
            ExpertState.COLD,
            out,
        )

        assert "[COLD]" in out.getvalue()

    def test_expert_bar_ok(self, reporter):

        out = io.StringIO()

        reporter._render_expert_bar(
            0,
            0.25,
            100,
            None,
            out,
        )

        assert "[OK]" in out.getvalue()

    def test_expert_bar_low(self, reporter):

        out = io.StringIO()

        reporter._render_expert_bar(
            0,
            0.0001,
            1,
            None,
            out,
        )

        assert "[LOW]" in out.getvalue()

    # =========================================================================
    # Histogram
    # =========================================================================

    def test_large_histogram_abbreviation(self, reporter):

        summary = SimpleNamespace(
            load_imbalance_score=1.0,
            event_count=100,
            n_experts=64,
            utilization=[0.1] * 64,
            token_counts=[10] * 64,
            total_tokens=640,
            max_util=0.1,
            min_util=0.1,
            mean_util=0.1,
        )

        out = io.StringIO()

        reporter._render_layer_histogram(
            "layer",
            summary,
            None,
            out,
        )

        assert "hidden" in out.getvalue()

    def test_large_histogram_problematic_expert(self, reporter):

        summary = SimpleNamespace(
            load_imbalance_score=1.0,
            event_count=100,
            n_experts=64,
            utilization=[0.1] * 64,
            token_counts=[10] * 64,
            total_tokens=640,
            max_util=0.1,
            min_util=0.1,
            mean_util=0.1,
        )

        collapse = SimpleNamespace(
            is_empty=False,
            experts=[
                SimpleNamespace(
                    expert_idx=5,
                    is_problematic=True,
                )
            ],
            expert=lambda idx: None,
        )

        out = io.StringIO()

        reporter._render_layer_histogram(
            "layer",
            summary,
            collapse,
            out,
        )

        assert "Layer total" in out.getvalue()

    # =========================================================================
    # Dead experts
    # =========================================================================

    def test_dead_expert_section_empty(
        self,
        reporter,
        fake_report,
    ):

        out = io.StringIO()

        reporter._render_dead_experts_section(
            fake_report,
            out,
        )

        assert out.getvalue() == ""

    def test_dead_expert_row(self, reporter, fake_report):

        dead = SimpleNamespace(
            layer_name="router.layer0",
            expert_idx=3,
            utilization_pct=0.0,
            consecutive_cold_steps=100,
            is_dead=True,
            is_cold=False,
            state=SimpleNamespace(value="DEAD"),
        )

        fake_report.dead_experts = lambda include_cold=True: [dead]

        out = io.StringIO()

        reporter._render_dead_experts_section(
            fake_report,
            out,
        )

        assert "DEAD" in out.getvalue()

    def test_cold_expert_row(self, reporter, fake_report):

        cold = SimpleNamespace(
            layer_name="router.layer0",
            expert_idx=1,
            utilization_pct=0.1,
            consecutive_cold_steps=10,
            is_dead=False,
            is_cold=True,
            state=SimpleNamespace(value="COLD"),
        )

        fake_report.dead_experts = lambda include_cold=True: [cold]

        out = io.StringIO()

        reporter._render_dead_experts_section(
            fake_report,
            out,
        )

        assert "COLD" in out.getvalue()

    # =========================================================================
    # Recommendations
    # =========================================================================

    def test_recommendations_empty(
        self,
        reporter,
        fake_report,
    ):

        fake_report.recommendations = lambda: []

        out = io.StringIO()

        reporter._render_recommendations_section(
            fake_report,
            out,
        )

        assert out.getvalue() == ""

    def test_recommendations_healthy(
        self,
        reporter,
        fake_report,
    ):

        out = io.StringIO()

        reporter._render_recommendations_section(
            fake_report,
            out,
        )

        assert "No action required" in out.getvalue()

    def test_recommendations_warn(
        self,
        reporter,
        fake_report,
    ):

        fake_report.recommendations = lambda: [
            "[WARN] Entropy declining. Continue monitoring."
        ]

        out = io.StringIO()

        reporter._render_recommendations_section(
            fake_report,
            out,
        )

        assert "Entropy declining" in out.getvalue()

    def test_recommendations_critical(
        self,
        reporter,
        fake_report,
    ):

        fake_report.overall_health = OverallHealth.CRITICAL

        fake_report.recommendations = lambda: [
            "[CRITICAL] Expert collapse detected. Increase balancing."
        ]

        out = io.StringIO()

        reporter._render_recommendations_section(
            fake_report,
            out,
        )

        assert "Expert collapse detected" in out.getvalue()

    def test_recommendation_continuation(
        self,
        reporter,
        fake_report,
    ):

        fake_report.recommendations = lambda: [
            "[WARN] Entropy low. This is the second sentence that triggers wrapping."
        ]

        out = io.StringIO()

        reporter._render_recommendations_section(
            fake_report,
            out,
        )

        text = out.getvalue()

        assert "Entropy low" in text
        assert "second sentence" in text

    # =========================================================================
    # print_summary exception handlers
    # =========================================================================

    @pytest.mark.parametrize(
        "method",
        [
            "_render_header",
            "_render_entropy_section",
            "_render_utilization_section",
            "_render_dead_experts_section",
            "_render_recommendations_section",
            "_render_footer",
        ],
    )
    def test_print_summary_exception_paths(
        self,
        reporter,
        fake_report,
        monkeypatch,
        method,
    ):

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            reporter,
            method,
            boom,
        )

        reporter.print_summary(fake_report)

    def test_print_summary_success(
        self,
        reporter,
        fake_report,
    ):
        reporter.print_summary(fake_report)
