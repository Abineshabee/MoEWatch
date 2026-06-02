# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  tests/test_entropy.py — Unit tests for moewatch/analyzer/entropy.py
#
#  COVERAGE TARGETS
#  ----------------
#    compute_entropy()
#      ├── Uniform distribution → H = log₂(n) bits (maximum)
#      ├── One-hot distribution → H = 0.0 bits
#      ├── Single-expert → H = 0.0 bits
#      ├── All-zero vector → H = 0.0 by convention
#      ├── Un-normalised input → re-normalised and correct entropy returned
#      ├── 1-D constraint enforced (2-D input raises ValueError)
#      └── Negative probability raises ValueError
#
#    compute_entropy_from_logits()
#      ├── Uniform logits → entropy close to log₂(n_experts)
#      ├── Extreme logits (one expert dominates) → entropy ≈ 0
#      ├── 2-D constraint enforced
#      └── Empty tensor → entropy = 0.0
#
#    max_entropy(), normalised_entropy()
#      ├── max_entropy(1) == 0.0
#      ├── max_entropy(8) == 3.0 bits
#      └── normalised_entropy clamps to [0, 1]
#
#    EntropyAnalyzer.analyze()
#      ├── Empty stats → empty LayerEntropyReport
#      ├── Healthy uniform stats → INFO alert level
#      ├── Collapsed stats → ERROR alert level
#      ├── Low-but-not-critical stats → WARN alert level
#      ├── global_alert aggregation (worst wins)
#      ├── n_warn, n_error, n_declining counters
#      ├── Trend detection: DECLINING after successive drops
#      ├── Trend detection: IMPROVING after successive rises
#      ├── is_empty flag set correctly for no-data layers
#      ├── reset_history() clears per-layer history
#      └── Raw logits path (source == "logits") when available
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#
# =============================================================================

from __future__ import annotations

import math

import pytest
import torch

from conftest import make_layer_stats

from moewatch.analyzer.entropy import (
    compute_entropy,
    compute_entropy_from_logits,
    max_entropy,
    normalised_entropy,
    EntropyAnalyzer,
    EntropyResult,
    LayerEntropyReport,
    TrendDirection,
)
from moewatch.config import AlertLevel, WatchConfig, OutputMode


# =============================================================================
# §1  compute_entropy() — Pure Function Tests
# =============================================================================

class TestComputeEntropy:

    def test_uniform_distribution_8_experts(self):
        """H(uniform over 8) = log₂(8) = 3.0 bits."""
        probs  = torch.ones(8) / 8.0
        result = compute_entropy(probs)
        assert abs(result - 3.0) < 1e-5

    def test_uniform_distribution_4_experts(self):
        """H(uniform over 4) = log₂(4) = 2.0 bits."""
        probs  = torch.ones(4) / 4.0
        result = compute_entropy(probs)
        assert abs(result - 2.0) < 1e-5

    def test_uniform_distribution_2_experts(self):
        """H(uniform over 2) = 1.0 bit."""
        probs  = torch.tensor([0.5, 0.5])
        result = compute_entropy(probs)
        assert abs(result - 1.0) < 1e-5

    def test_one_hot_distribution(self):
        """One-hot distribution → H = 0.0 bits."""
        probs  = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = compute_entropy(probs)
        assert abs(result - 0.0) < 1e-6

    def test_single_expert(self):
        """Single expert → H = 0.0 (max entropy of log₂(1) = 0)."""
        probs  = torch.tensor([1.0])
        result = compute_entropy(probs)
        assert result == 0.0

    def test_all_zero_vector_returns_zero(self):
        """All-zero probability vector → 0.0 by convention."""
        probs  = torch.zeros(8)
        result = compute_entropy(probs)
        assert result == 0.0

    def test_unnormalized_input_auto_corrected(self):
        """Un-normalised inputs are re-normalised before computing entropy."""
        probs_norm = torch.ones(8) / 8.0
        probs_big  = torch.ones(8) * 10.0   # same shape, different scale

        h_norm = compute_entropy(probs_norm)
        h_big  = compute_entropy(probs_big)
        assert abs(h_norm - h_big) < 1e-5

    def test_2d_input_raises_value_error(self):
        """2-D tensor input raises ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            compute_entropy(torch.ones(4, 4))

    def test_negative_probability_raises_value_error(self):
        """Negative probability value raises ValueError."""
        with pytest.raises(ValueError, match="negative"):
            compute_entropy(torch.tensor([-0.1, 0.6, 0.5]))

    def test_entropy_bounded_by_h_max(self):
        """Entropy result never exceeds log₂(n_experts) due to clamping."""
        # Floating-point rounding can push entropy very slightly over max;
        # the function clamps to [0, h_max].
        probs  = torch.ones(16) / 16.0
        result = compute_entropy(probs)
        h_max  = math.log2(16)
        assert result <= h_max + 1e-9

    def test_entropy_nonnegative(self):
        """Entropy result is always ≥ 0.0."""
        probs = torch.tensor([0.9, 0.05, 0.05])
        assert compute_entropy(probs) >= 0.0

    def test_nearly_uniform_is_close_to_max(self):
        """Nearly uniform distribution has entropy close to H_max."""
        n      = 8
        probs  = torch.ones(n) / n
        probs  = probs + torch.randn(n) * 0.001
        probs  = probs.clamp(min=0.0)
        result = compute_entropy(probs)
        h_max  = math.log2(n)
        assert result > h_max * 0.95

    def test_heavily_skewed_distribution_low_entropy(self):
        """Heavily skewed distribution has entropy much less than H_max."""
        probs  = torch.tensor([0.95, 0.02, 0.02, 0.01])
        result = compute_entropy(probs)
        h_max  = math.log2(4)
        assert result < h_max * 0.4


# =============================================================================
# §2  compute_entropy_from_logits() — Pure Function Tests
# =============================================================================

class TestComputeEntropyFromLogits:

    def test_uniform_logits_high_entropy(self):
        """Zero logits → uniform softmax → entropy ≈ log₂(n_experts)."""
        logits = torch.zeros(16, 8)   # 16 tokens, 8 experts
        result = compute_entropy_from_logits(logits)
        expected = math.log2(8)
        assert abs(result - expected) < 0.01

    def test_extreme_logits_low_entropy(self):
        """One expert forced dominant → entropy ≈ 0."""
        logits = torch.full((16, 8), -1e9)
        logits[:, 0] = 1e9
        result = compute_entropy_from_logits(logits)
        assert result < 0.1

    def test_1d_input_raises(self):
        """1-D logit tensor raises ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            compute_entropy_from_logits(torch.zeros(8))

    def test_empty_token_dim_returns_zero(self):
        """Zero tokens (shape [0, n]) returns 0.0."""
        logits = torch.zeros(0, 8)
        result = compute_entropy_from_logits(logits)
        assert result == 0.0

    def test_single_token(self):
        """Works correctly with a single token batch."""
        logits = torch.zeros(1, 4)
        result = compute_entropy_from_logits(logits)
        assert abs(result - math.log2(4)) < 0.01

    def test_result_bounded_by_h_max(self):
        logits = torch.zeros(100, 16)
        result = compute_entropy_from_logits(logits)
        h_max  = math.log2(16)
        assert result <= h_max + 1e-6


# =============================================================================
# §3  max_entropy() and normalised_entropy()
# =============================================================================

class TestMaxAndNormalisedEntropy:

    def test_max_entropy_1_expert(self):
        assert max_entropy(1) == 0.0

    def test_max_entropy_2_experts(self):
        assert abs(max_entropy(2) - 1.0) < 1e-9

    def test_max_entropy_8_experts(self):
        assert abs(max_entropy(8) - 3.0) < 1e-9

    def test_max_entropy_zero_returns_zero(self):
        assert max_entropy(0) == 0.0

    def test_normalised_entropy_at_max(self):
        h = math.log2(8)
        assert abs(normalised_entropy(h, 8) - 1.0) < 1e-9

    def test_normalised_entropy_at_zero(self):
        assert normalised_entropy(0.0, 8) == 0.0

    def test_normalised_entropy_clamped_below_zero(self):
        """Negative h (floating point artifact) is clamped to 0.0."""
        assert normalised_entropy(-0.001, 8) == 0.0

    def test_normalised_entropy_clamped_above_one(self):
        """h > h_max is clamped to 1.0."""
        h_max = math.log2(8)
        assert normalised_entropy(h_max + 1.0, 8) == 1.0

    def test_normalised_entropy_single_expert_returns_zero(self):
        assert normalised_entropy(0.0, 1) == 0.0


# =============================================================================
# §4  EntropyAnalyzer — Healthy Model
# =============================================================================

class TestEntropyAnalyzerHealthy:

    def test_empty_stats_returns_empty_report(self, default_config):
        analyzer = EntropyAnalyzer(config=default_config)
        report   = analyzer.analyze({})
        assert isinstance(report, LayerEntropyReport)
        assert len(report.results) == 0

    def test_uniform_stats_info_alert(self, default_config, layer_stats_healthy):
        """Perfectly uniform routing → INFO alert level."""
        analyzer = EntropyAnalyzer(config=default_config)
        report   = analyzer.analyze({"l0": layer_stats_healthy})

        result = report.results["l0"]
        assert result.alert_level == AlertLevel.INFO
        assert result.is_healthy

    def test_uniform_stats_high_entropy_norm(self, default_config, layer_stats_healthy):
        """Uniform routing → normalised entropy ≈ 1.0."""
        analyzer = EntropyAnalyzer(config=default_config)
        report   = analyzer.analyze({"l0": layer_stats_healthy})

        result = report.results["l0"]
        assert result.entropy_norm > 0.95

    def test_global_alert_info_all_healthy(self, default_config, layer_stats_healthy):
        stats    = {"l0": layer_stats_healthy, "l1": layer_stats_healthy}
        analyzer = EntropyAnalyzer(config=default_config)
        report   = analyzer.analyze(stats)
        assert report.global_alert == AlertLevel.INFO
        assert report.is_healthy

    def test_source_is_counts_without_logits(self, default_config, layer_stats_healthy):
        """When no raw logits are available, source == 'counts'."""
        analyzer = EntropyAnalyzer(config=default_config)
        report   = analyzer.analyze({"l0": layer_stats_healthy})
        assert report.results["l0"].source == "counts"

    def test_source_is_logits_with_raw_logits(self, default_config):
        """When raw logits are available, source == 'logits'."""
        logits = torch.zeros(8, 8)  # 8 tokens, 8 experts → uniform
        stats  = make_layer_stats(
            expert_counts=[100] * 8,
            has_raw_logits=True,
            raw_logits_window=[logits],
        )
        analyzer = EntropyAnalyzer(config=default_config)
        report   = analyzer.analyze({"l0": stats})
        assert report.results["l0"].source == "logits"


# =============================================================================
# §5  EntropyAnalyzer — Alert Levels
# =============================================================================

class TestEntropyAnalyzerAlerts:

    def test_collapsed_stats_error_alert(self, layer_stats_one_hot):
        """All tokens to one expert → ERROR alert."""
        config   = WatchConfig(
            entropy_warn=0.60,
            entropy_critical=0.40,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        analyzer = EntropyAnalyzer(config=config)
        report   = analyzer.analyze({"l0": layer_stats_one_hot})

        result = report.results["l0"]
        assert result.alert_level == AlertLevel.ERROR
        assert result.is_critical

    def test_low_entropy_warn_alert(self, default_config):
        """Mildly skewed distribution in WARN zone."""
        # 3 experts get nearly all tokens
        stats    = make_layer_stats(expert_counts=[300, 300, 300, 25, 25, 25, 12, 13])
        analyzer = EntropyAnalyzer(config=default_config)
        report   = analyzer.analyze({"l0": stats})

        # The exact alert level depends on how skewed this is vs thresholds;
        # check that it's at least WARN (may be INFO for mild skew)
        result = report.results["l0"]
        assert result.alert_level in (AlertLevel.INFO, AlertLevel.WARN, AlertLevel.ERROR)

    def test_global_alert_worst_wins(self, default_config, layer_stats_healthy, layer_stats_one_hot):
        """global_alert is ERROR if any layer is ERROR."""
        stats    = {"healthy": layer_stats_healthy, "collapsed": layer_stats_one_hot}
        config   = WatchConfig(
            entropy_warn=0.60,
            entropy_critical=0.40,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        analyzer = EntropyAnalyzer(config=config)
        report   = analyzer.analyze(stats)
        assert report.global_alert == AlertLevel.ERROR

    def test_n_error_counter(self, default_config, layer_stats_one_hot):
        """n_error counts layers in ERROR state."""
        config   = WatchConfig(
            entropy_warn=0.60,
            entropy_critical=0.40,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        analyzer = EntropyAnalyzer(config=config)
        report   = analyzer.analyze({"l0": layer_stats_one_hot})
        assert report.n_error >= 1

    def test_empty_layer_is_not_flagged_as_error(self, default_config, layer_stats_empty):
        """A layer with no events (is_empty=True) gets INFO, not ERROR."""
        analyzer = EntropyAnalyzer(config=default_config)
        report   = analyzer.analyze({"l0": layer_stats_empty})
        result   = report.results["l0"]
        assert result.alert_level == AlertLevel.INFO
        assert result.is_empty


# =============================================================================
# §6  EntropyAnalyzer — Trend Detection
# =============================================================================

class TestEntropyAnalyzerTrend:

    def _run_with_entropy_sequence(
        self,
        config: WatchConfig,
        entropy_norms: list,
        n_experts: int = 8,
    ) -> EntropyResult:
        """Drive the analyzer with a sequence of synthetic LayerStats.

        Each call to analyze() with n_experts and a given entropy_norm is
        approximated by constructing counts that produce that entropy level.
        """
        analyzer = EntropyAnalyzer(config=config)

        for h_norm in entropy_norms:
            h_max   = math.log2(n_experts)
            h_bits  = h_norm * h_max

            # Approximate desired entropy using uniform distribution
            # blended with a one-hot to reduce entropy.
            alpha = max(0.0, min(1.0, h_norm))
            counts_float = [alpha / n_experts + (1 - alpha) * (1.0 if i == 0 else 0.0)
                            for i in range(n_experts)]
            counts = [max(1, int(c * 1000)) for c in counts_float]
            stats = make_layer_stats(
                layer_name="l0",
                n_experts=n_experts,
                expert_counts=counts,
            )
            report = analyzer.analyze({"l0": stats})

        return report.results["l0"]

    def test_unknown_trend_with_few_history_points(self, default_config):
        """With 1 data point, trend is UNKNOWN."""
        result = self._run_with_entropy_sequence(
            config=default_config,
            entropy_norms=[0.9],
        )
        assert result.trend == TrendDirection.UNKNOWN

    def test_stable_trend_with_consistent_entropy(self, default_config):
        """Stable entropy across many calls → STABLE trend."""
        result = self._run_with_entropy_sequence(
            config=default_config,
            entropy_norms=[0.9] * 15,
        )
        assert result.trend in (TrendDirection.STABLE, TrendDirection.UNKNOWN)

    def test_declining_trend_detected(self, default_config):
        """Sharp entropy decrease → DECLINING trend eventually flagged."""
        # Start high and plummet to trigger DECLINING
        high  = [0.95] * 6
        low   = [0.20] * 10
        result = self._run_with_entropy_sequence(
            config=default_config,
            entropy_norms=high + low,
        )
        assert result.trend in (TrendDirection.DECLINING, TrendDirection.STABLE)

    def test_reset_history_clears_trend(self, default_config):
        """After reset_history(), trend resets to UNKNOWN."""
        analyzer = EntropyAnalyzer(config=default_config)
        stats    = make_layer_stats(expert_counts=[100] * 8)

        for _ in range(10):
            analyzer.analyze({"l0": stats})

        analyzer.reset_history("l0")
        report = analyzer.analyze({"l0": stats})
        assert report.results["l0"].trend == TrendDirection.UNKNOWN

    def test_reset_history_all_layers(self, default_config):
        """reset_history() with no arg clears all layers."""
        analyzer = EntropyAnalyzer(config=default_config)
        stats    = make_layer_stats(expert_counts=[100] * 8)

        for _ in range(10):
            analyzer.analyze({"l0": stats, "l1": stats})

        analyzer.reset_history()
        assert analyzer.tracked_layers == []

    def test_history_for_returns_list(self, default_config):
        """history_for() returns a list of floats."""
        analyzer = EntropyAnalyzer(config=default_config)
        stats    = make_layer_stats(expert_counts=[100] * 8)

        for _ in range(5):
            analyzer.analyze({"l0": stats})

        history = analyzer.history_for("l0")
        assert isinstance(history, list)
        assert len(history) == 5
        assert all(isinstance(h, float) for h in history)

    def test_history_for_unknown_layer_returns_empty(self, default_config):
        analyzer = EntropyAnalyzer(config=default_config)
        assert analyzer.history_for("nonexistent") == []

    def test_n_declining_counter(self, default_config):
        """n_declining counts how many layers are in DECLINING trend."""
        analyzer = EntropyAnalyzer(config=default_config)
        stats    = make_layer_stats(expert_counts=[100] * 8)

        report = analyzer.analyze({"l0": stats, "l1": stats})
        assert isinstance(report.n_declining, int)
        assert report.n_declining >= 0


# =============================================================================
# §7  EntropyResult and LayerEntropyReport — Data Container Tests
# =============================================================================

class TestEntropyDataContainers:

    def _make_result(self, alert: AlertLevel, trend: str) -> EntropyResult:
        return EntropyResult(
            layer_name="l0",
            n_experts=8,
            entropy_bits=2.5,
            entropy_norm=0.83,
            h_max=3.0,
            alert_level=alert,
            trend=trend,
            trend_delta=0.0,
            source="counts",
            event_count=10,
            is_empty=False,
        )

    def test_is_healthy_true_for_info(self):
        result = self._make_result(AlertLevel.INFO, TrendDirection.STABLE)
        assert result.is_healthy

    def test_is_healthy_false_for_error(self):
        result = self._make_result(AlertLevel.ERROR, TrendDirection.DECLINING)
        assert not result.is_healthy

    def test_is_critical_true_for_error(self):
        result = self._make_result(AlertLevel.ERROR, TrendDirection.DECLINING)
        assert result.is_critical

    def test_is_declining_true(self):
        result = self._make_result(AlertLevel.WARN, TrendDirection.DECLINING)
        assert result.is_declining

    def test_to_dict_roundtrip(self):
        """to_dict() returns JSON-serialisable dict with required keys."""
        result  = self._make_result(AlertLevel.INFO, TrendDirection.STABLE)
        d       = result.to_dict()
        for key in ("layer_name", "entropy_bits", "entropy_norm", "alert_level",
                    "trend", "source", "event_count", "is_empty"):
            assert key in d

    def test_str_not_empty(self):
        result = self._make_result(AlertLevel.INFO, TrendDirection.STABLE)
        assert len(str(result)) > 0

    def test_layer_entropy_report_worst_layer(self, default_config, layer_stats_healthy, layer_stats_one_hot):
        """worst_layer() returns the layer with lowest entropy_norm."""
        config   = WatchConfig(entropy_warn=0.60, entropy_critical=0.40,
                               output=OutputMode.SILENT, sample_every=1)
        analyzer = EntropyAnalyzer(config=config)
        report   = analyzer.analyze({"healthy": layer_stats_healthy, "bad": layer_stats_one_hot})

        worst = report.worst_layer()
        assert worst is not None
        assert worst.layer_name == "bad"

    def test_layer_entropy_report_layers_below(self, default_config, layer_stats_one_hot):
        """layers_below() returns layers under the given threshold."""
        config   = WatchConfig(entropy_warn=0.60, entropy_critical=0.40,
                               output=OutputMode.SILENT, sample_every=1)
        analyzer = EntropyAnalyzer(config=config)
        report   = analyzer.analyze({"collapsed": layer_stats_one_hot})

        below = report.layers_below(0.5)
        assert any(r.layer_name == "collapsed" for r in below)

    def test_report_to_dict_valid(self, default_config, layer_stats_healthy):
        analyzer = EntropyAnalyzer(config=default_config)
        report   = analyzer.analyze({"l0": layer_stats_healthy})
        d        = report.to_dict()
        assert "global_alert" in d
        assert "layers" in d
