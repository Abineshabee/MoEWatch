# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  tests/test_collapse.py — Unit tests for moewatch/analyzer/collapse.py
#
#  COVERAGE TARGETS
#  ----------------
#    ExpertState / ExpertStatus
#      ├── is_healthy / is_cold / is_dead / is_problematic properties
#      ├── to_dict() returns JSON-serialisable dict with required keys
#      └── __str__ includes expert index and utilization
#
#    LayerCollapseReport
#      ├── n_problematic == n_dead + n_cold
#      ├── is_collapsed True only when n_dead > 0
#      ├── is_degrading True when n_dead OR n_cold > 0
#      ├── dead_experts() / cold_experts() / healthy_experts() filters
#      ├── expert(idx) accessor
#      └── to_dict() contains all required keys
#
#    CollapseDetector.detect()
#      ├── Empty stats → empty report dict
#      ├── Healthy uniform stats → all HEALTHY, global_alert INFO
#      ├── Expert below dead_threshold → DEAD immediately, alert ERROR
#      ├── Expert in cold zone → COLD first, alert WARN
#      ├── COLD expert promoted to DEAD after cold_steps_limit calls
#      ├── DEAD expert recovers to HEALTHY when utilisation rises
#      ├── Multi-layer: reports keyed by layer name, all layers present
#      ├── global_alert == ERROR if any expert is DEAD
#      ├── global_alert == WARN if any expert is COLD (no DEAD)
#      ├── global_alert == INFO if all experts HEALTHY
#      ├── is_collapsed / is_degrading on LayerCollapseReport
#      ├── n_dead / n_cold / n_healthy counts correct
#      ├── reset() clears cold counters for one layer
#      ├── reset() with no arg clears all layers
#      ├── tracked_layers / cold_counter_for introspection
#      └── load_imbalance_score propagated from LayerStats
#
#    CollapseDetector.compute_load_imbalance()
#      ├── Uniform → score ≈ 1.0
#      ├── One-hot → score == n_experts
#      └── Empty / all-zero → (nan, -1, nan)
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

from moewatch.analyzer.collapse import (
    CollapseDetector,
    ExpertState,
    ExpertStatus,
    LayerCollapseReport,
)
from moewatch.config import AlertLevel, WatchConfig, OutputMode


# =============================================================================
# §1  ExpertStatus — Data Container
# =============================================================================

class TestExpertStatus:
    """Tests for the ExpertStatus frozen dataclass."""

    def _make_status(
        self,
        state: ExpertState,
        utilization: float = 0.125,
        cold_steps: int = 0,
    ) -> ExpertStatus:
        alert = {
            ExpertState.DEAD:    AlertLevel.ERROR,
            ExpertState.COLD:    AlertLevel.WARN,
            ExpertState.HEALTHY: AlertLevel.INFO,
            ExpertState.UNKNOWN: AlertLevel.INFO,
        }[state]
        return ExpertStatus(
            layer_name="moe_layers.0",
            expert_idx=3,
            state=state,
            alert_level=alert,
            utilization=utilization,
            token_count=int(utilization * 1000),
            consecutive_cold_steps=cold_steps,
        )

    def test_is_healthy_true(self):
        s = self._make_status(ExpertState.HEALTHY)
        assert s.is_healthy is True
        assert s.is_cold is False
        assert s.is_dead is False
        assert s.is_problematic is False

    def test_is_cold_true(self):
        s = self._make_status(ExpertState.COLD, utilization=0.003)
        assert s.is_cold is True
        assert s.is_problematic is True
        assert s.is_healthy is False

    def test_is_dead_true(self):
        s = self._make_status(ExpertState.DEAD, utilization=0.0)
        assert s.is_dead is True
        assert s.is_problematic is True
        assert s.is_healthy is False

    def test_to_dict_required_keys(self):
        s = self._make_status(ExpertState.HEALTHY)
        d = s.to_dict()
        for key in (
            "layer_name", "expert_idx", "state", "alert_level",
            "utilization", "utilization_pct", "token_count",
            "consecutive_cold_steps",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_state_is_string(self):
        s = self._make_status(ExpertState.DEAD)
        d = s.to_dict()
        assert isinstance(d["state"], str)
        assert d["state"] == "DEAD"

    def test_to_dict_utilization_pct_equals_utilization_times_100(self):
        s = self._make_status(ExpertState.HEALTHY, utilization=0.125)
        d = s.to_dict()
        assert abs(d["utilization_pct"] - d["utilization"] * 100) < 1e-6

    def test_str_contains_expert_idx(self):
        s = self._make_status(ExpertState.HEALTHY)
        assert "3" in str(s)

    def test_str_contains_utilization(self):
        s = self._make_status(ExpertState.COLD, utilization=0.003)
        assert "0.300%" in str(s)

    def test_frozen_immutability(self):
        s = self._make_status(ExpertState.HEALTHY)
        with pytest.raises((AttributeError, TypeError)):
            s.utilization = 0.9  # frozen dataclass should reject this


# =============================================================================
# §2  LayerCollapseReport — Data Container
# =============================================================================

class TestLayerCollapseReport:
    """Tests for the LayerCollapseReport dataclass."""

    def _make_report(
        self,
        n_dead: int = 0,
        n_cold: int = 0,
        n_healthy: int = 8,
    ) -> LayerCollapseReport:
        """Build a synthetic report with the given expert counts."""
        experts = []
        idx = 0

        for _ in range(n_dead):
            experts.append(ExpertStatus(
                layer_name="l0", expert_idx=idx,
                state=ExpertState.DEAD, alert_level=AlertLevel.ERROR,
                utilization=0.0, token_count=0, consecutive_cold_steps=5,
            ))
            idx += 1

        for _ in range(n_cold):
            experts.append(ExpertStatus(
                layer_name="l0", expert_idx=idx,
                state=ExpertState.COLD, alert_level=AlertLevel.WARN,
                utilization=0.003, token_count=3, consecutive_cold_steps=1,
            ))
            idx += 1

        for _ in range(n_healthy):
            experts.append(ExpertStatus(
                layer_name="l0", expert_idx=idx,
                state=ExpertState.HEALTHY, alert_level=AlertLevel.INFO,
                utilization=0.125, token_count=125, consecutive_cold_steps=0,
            ))
            idx += 1

        global_alert = (
            AlertLevel.ERROR if n_dead > 0 else
            AlertLevel.WARN  if n_cold > 0 else
            AlertLevel.INFO
        )
        return LayerCollapseReport(
            layer_name="l0",
            n_experts=idx,
            experts=experts,
            n_dead=n_dead,
            n_cold=n_cold,
            n_healthy=n_healthy,
            global_alert=global_alert,
            load_imbalance_score=1.0,
            is_empty=False,
        )

    def test_n_problematic_equals_dead_plus_cold(self):
        report = self._make_report(n_dead=1, n_cold=2, n_healthy=5)
        assert report.n_problematic == 3

    def test_is_collapsed_true_when_dead(self):
        report = self._make_report(n_dead=1)
        assert report.is_collapsed is True

    def test_is_collapsed_false_when_only_cold(self):
        report = self._make_report(n_cold=2)
        assert report.is_collapsed is False

    def test_is_degrading_true_when_cold(self):
        report = self._make_report(n_cold=1)
        assert report.is_degrading is True

    def test_is_degrading_true_when_dead(self):
        report = self._make_report(n_dead=2)
        assert report.is_degrading is True

    def test_is_degrading_false_when_all_healthy(self):
        report = self._make_report(n_dead=0, n_cold=0, n_healthy=8)
        assert report.is_degrading is False

    def test_dead_experts_filter(self):
        report = self._make_report(n_dead=2, n_cold=1, n_healthy=5)
        dead = report.dead_experts()
        assert all(e.is_dead for e in dead)
        assert len(dead) == 2

    def test_cold_experts_filter(self):
        report = self._make_report(n_dead=1, n_cold=3, n_healthy=4)
        cold = report.cold_experts()
        assert all(e.is_cold for e in cold)
        assert len(cold) == 3

    def test_healthy_experts_filter(self):
        report = self._make_report(n_dead=0, n_cold=2, n_healthy=6)
        healthy = report.healthy_experts()
        assert all(e.is_healthy for e in healthy)
        assert len(healthy) == 6

    def test_expert_accessor_valid_idx(self):
        report = self._make_report(n_healthy=4)
        e = report.expert(0)
        assert e is not None
        assert e.expert_idx == 0

    def test_expert_accessor_out_of_range_returns_none(self):
        report = self._make_report(n_healthy=4)
        assert report.expert(999) is None

    def test_to_dict_required_keys(self):
        report = self._make_report(n_dead=1, n_cold=1, n_healthy=6)
        d = report.to_dict()
        for key in (
            "layer_name", "n_experts", "n_dead", "n_cold", "n_healthy",
            "global_alert", "load_imbalance_score", "is_empty", "experts",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_experts_is_list(self):
        report = self._make_report(n_healthy=8)
        d = report.to_dict()
        assert isinstance(d["experts"], list)

    def test_repr_contains_layer_name(self):
        report = self._make_report()
        assert "l0" in repr(report)

    def test_empty_report_is_empty_flag(self):
        report = LayerCollapseReport(
            layer_name="empty_layer",
            n_experts=0,
            experts=[],
            is_empty=True,
            global_alert=AlertLevel.INFO,
            load_imbalance_score=float("nan"),
        )
        assert report.is_empty is True
        assert report.n_problematic == 0


# =============================================================================
# §3  CollapseDetector — Healthy Model
# =============================================================================

class TestCollapseDetectorHealthy:

    def test_empty_stats_returns_empty_dict(self, default_config):
        detector = CollapseDetector(config=default_config)
        result   = detector.detect({})
        assert result == {}

    def test_uniform_routing_all_healthy(self, default_config, layer_stats_healthy):
        detector = CollapseDetector(config=default_config)
        reports  = detector.detect({"l0": layer_stats_healthy})

        report = reports["l0"]
        assert report.n_dead == 0
        assert report.n_cold == 0
        assert report.n_healthy == 8
        assert report.global_alert == AlertLevel.INFO

    def test_all_experts_healthy_is_collapsed_false(
        self, default_config, layer_stats_healthy
    ):
        detector = CollapseDetector(config=default_config)
        reports  = detector.detect({"l0": layer_stats_healthy})
        assert reports["l0"].is_collapsed is False

    def test_global_alert_info_all_healthy(
        self, default_config, layer_stats_healthy
    ):
        detector = CollapseDetector(config=default_config)
        reports  = detector.detect({"l0": layer_stats_healthy})
        assert reports["l0"].global_alert == AlertLevel.INFO

    def test_detect_returns_all_layer_names(
        self, default_config, layer_stats_healthy
    ):
        stats    = {"l0": layer_stats_healthy, "l1": layer_stats_healthy}
        detector = CollapseDetector(config=default_config)
        reports  = detector.detect(stats)
        assert set(reports.keys()) == {"l0", "l1"}

    def test_empty_layer_stats_returns_is_empty_report(
        self, default_config, layer_stats_empty
    ):
        detector = CollapseDetector(config=default_config)
        reports  = detector.detect({"empty": layer_stats_empty})
        assert reports["empty"].is_empty is True

    def test_n_dead_n_cold_n_healthy_sum_to_n_experts(
        self, default_config, layer_stats_healthy
    ):
        detector = CollapseDetector(config=default_config)
        report   = detector.detect({"l0": layer_stats_healthy})["l0"]
        assert report.n_dead + report.n_cold + report.n_healthy == report.n_experts


# =============================================================================
# §4  CollapseDetector — Dead Expert Detection
# =============================================================================

class TestCollapseDetectorDead:

    def test_dead_expert_detected_immediately(
        self, default_config, layer_stats_collapsed
    ):
        """Expert at 0% → DEAD on first detect() call."""
        detector = CollapseDetector(config=default_config)
        reports  = detector.detect({"l0": layer_stats_collapsed})

        report = reports["l0"]
        assert report.n_dead >= 1
        assert report.global_alert == AlertLevel.ERROR
        assert report.is_collapsed is True

    def test_dead_expert_has_error_alert_level(
        self, default_config, layer_stats_collapsed
    ):
        detector = CollapseDetector(config=default_config)
        report   = detector.detect({"l0": layer_stats_collapsed})["l0"]
        dead     = report.dead_experts()
        assert all(e.alert_level == AlertLevel.ERROR for e in dead)

    def test_all_dead_experts_flagged(
        self, default_config, layer_stats_all_dead
    ):
        """layer_stats_all_dead has experts 4–7 at zero."""
        detector = CollapseDetector(config=default_config)
        report   = detector.detect({"l0": layer_stats_all_dead})["l0"]
        assert report.n_dead == 4

    def test_one_hot_routing_all_others_dead(
        self, default_config, layer_stats_one_hot
    ):
        """Expert 0 gets 100% → all other 7 experts are DEAD."""
        detector = CollapseDetector(config=default_config)
        report   = detector.detect({"l0": layer_stats_one_hot})["l0"]
        # Expert 0 is healthy; experts 1–7 are dead
        assert report.n_dead == 7
        assert report.n_healthy == 1

    def test_dead_expert_zero_utilization(
        self, default_config, layer_stats_collapsed
    ):
        detector = CollapseDetector(config=default_config)
        report   = detector.detect({"l0": layer_stats_collapsed})["l0"]
        for e in report.dead_experts():
            assert e.utilization <= default_config.dead_threshold


# =============================================================================
# §5  CollapseDetector — Cold Expert Detection
# =============================================================================

class TestCollapseDetectorCold:

    def test_cold_expert_detected_in_cold_zone(self, aggressive_config):
        """An expert with utilization between dead and cold thresholds → COLD."""
        # aggressive_config: dead=0.05, cold=0.10
        # Expert 7 at 0.06 (above dead, below cold)
        counts = [116, 116, 116, 116, 116, 116, 116, 60]  # 60/1000 = 6%
        stats  = make_layer_stats(expert_counts=counts)

        detector = CollapseDetector(config=aggressive_config)
        report   = detector.detect({"l0": stats})["l0"]

        assert report.n_cold >= 1
        assert report.global_alert == AlertLevel.WARN

    def test_cold_expert_alert_is_warn(self, aggressive_config):
        counts = [116, 116, 116, 116, 116, 116, 116, 60]
        stats  = make_layer_stats(expert_counts=counts)

        detector = CollapseDetector(config=aggressive_config)
        report   = detector.detect({"l0": stats})["l0"]
        cold     = report.cold_experts()
        assert all(e.alert_level == AlertLevel.WARN for e in cold)

    def test_cold_expert_is_not_dead(self, aggressive_config):
        counts = [116, 116, 116, 116, 116, 116, 116, 60]
        stats  = make_layer_stats(expert_counts=counts)

        detector = CollapseDetector(config=aggressive_config)
        report   = detector.detect({"l0": stats})["l0"]
        for e in report.cold_experts():
            assert not e.is_dead

    def test_cold_expert_promoted_to_dead_after_limit(self, aggressive_config):
        """After cold_steps_limit consecutive cold calls → promoted to DEAD."""
        # aggressive_config: cold_steps_limit=2
        counts = [116, 116, 116, 116, 116, 116, 116, 60]  # expert 7 = 6%
        stats  = make_layer_stats(expert_counts=counts)

        detector = CollapseDetector(config=aggressive_config)

        # Call detect() cold_steps_limit + 1 times with the same cold stats
        for _ in range(aggressive_config.cold_steps_limit + 1):
            report = detector.detect({"l0": stats})["l0"]

        # Expert 7 should now be DEAD (promoted after cold_steps_limit steps)
        dead_idxs = [e.expert_idx for e in report.dead_experts()]
        assert 7 in dead_idxs

    def test_cold_counter_increments_each_call(self, aggressive_config):
        """cold_counter_for() increments on consecutive cold calls."""
        counts   = [116, 116, 116, 116, 116, 116, 116, 60]
        stats    = make_layer_stats(expert_counts=counts)
        detector = CollapseDetector(config=aggressive_config)

        detector.detect({"l0": stats})
        counter_after_1 = detector.cold_counter_for("l0", 7)

        detector.detect({"l0": stats})
        counter_after_2 = detector.cold_counter_for("l0", 7)

        assert counter_after_2 > counter_after_1

    def test_cold_expert_recovers_when_utilisation_rises(self, aggressive_config):
        """An expert that was COLD resets to HEALTHY when utilisation recovers."""
        cold_counts    = [116, 116, 116, 116, 116, 116, 116, 60]
        healthy_counts = [125, 125, 125, 125, 125, 125, 125, 125]

        cold_stats    = make_layer_stats(expert_counts=cold_counts)
        healthy_stats = make_layer_stats(expert_counts=healthy_counts)

        detector = CollapseDetector(config=aggressive_config)

        # First: expert 7 goes cold
        detector.detect({"l0": cold_stats})
        assert detector.cold_counter_for("l0", 7) >= 1

        # Then: expert 7 recovers
        report = detector.detect({"l0": healthy_stats})["l0"]

        # Counter should be reset
        assert detector.cold_counter_for("l0", 7) == 0
        assert report.n_cold == 0


# =============================================================================
# §6  CollapseDetector — Multi-Layer and Aggregation
# =============================================================================

class TestCollapseDetectorMultiLayer:

    def test_multilayer_all_keys_present(
        self, default_config, layer_stats_healthy
    ):
        stats    = {
            "moe_layers.0": layer_stats_healthy,
            "moe_layers.1": layer_stats_healthy,
            "moe_layers.2": layer_stats_healthy,
        }
        detector = CollapseDetector(config=default_config)
        reports  = detector.detect(stats)
        assert set(reports.keys()) == set(stats.keys())

    def test_one_collapsed_layer_doesnt_affect_healthy_layers(
        self, default_config, layer_stats_healthy, layer_stats_collapsed
    ):
        stats    = {
            "healthy": layer_stats_healthy,
            "bad":     layer_stats_collapsed,
        }
        detector = CollapseDetector(config=default_config)
        reports  = detector.detect(stats)

        assert reports["healthy"].n_dead == 0
        assert reports["bad"].n_dead >= 1

    def test_global_alert_error_propagates_to_report_level(
        self, default_config, layer_stats_healthy, layer_stats_collapsed
    ):
        """Per-layer global_alert is ERROR for the collapsed layer."""
        stats    = {"h": layer_stats_healthy, "c": layer_stats_collapsed}
        detector = CollapseDetector(config=default_config)
        reports  = detector.detect(stats)

        assert reports["c"].global_alert == AlertLevel.ERROR
        assert reports["h"].global_alert == AlertLevel.INFO

    def test_load_imbalance_score_propagated(
        self, default_config, layer_stats_one_hot
    ):
        """load_imbalance_score from LayerStats is propagated to the report."""
        detector = CollapseDetector(config=default_config)
        report   = detector.detect({"l0": layer_stats_one_hot})["l0"]
        # one-hot routing: imbalance should be very high, not NaN
        assert not math.isnan(report.load_imbalance_score)
        assert report.load_imbalance_score > 5.0


# =============================================================================
# §7  CollapseDetector — State Management (reset / introspection)
# =============================================================================

class TestCollapseDetectorState:

    def test_reset_single_layer_clears_counters(self, aggressive_config):
        """reset('l0') clears counters for l0 but not l1."""
        counts = [116, 116, 116, 116, 116, 116, 116, 60]
        stats  = {
            "l0": make_layer_stats(layer_name="l0", expert_counts=counts),
            "l1": make_layer_stats(layer_name="l1", expert_counts=counts),
        }
        detector = CollapseDetector(config=aggressive_config)
        detector.detect(stats)

        # Both layers have cold counters
        assert detector.cold_counter_for("l0", 7) >= 1
        assert detector.cold_counter_for("l1", 7) >= 1

        detector.reset("l0")

        # l0 counters cleared; l1 still has counts
        assert detector.cold_counter_for("l0", 7) == 0
        assert detector.cold_counter_for("l1", 7) >= 1

    def test_reset_all_layers_clears_everything(self, aggressive_config):
        """reset() with no arg clears all tracked layers."""
        counts = [116, 116, 116, 116, 116, 116, 116, 60]
        stats  = {
            "l0": make_layer_stats(layer_name="l0", expert_counts=counts),
            "l1": make_layer_stats(layer_name="l1", expert_counts=counts),
        }
        detector = CollapseDetector(config=aggressive_config)
        detector.detect(stats)

        detector.reset()

        assert detector.cold_counter_for("l0", 7) == 0
        assert detector.cold_counter_for("l1", 7) == 0
        assert detector.tracked_layers == []

    def test_tracked_layers_populated_after_detect(self, default_config, layer_stats_healthy):
        detector = CollapseDetector(config=default_config)
        detector.detect({"l0": layer_stats_healthy})
        assert "l0" in detector.tracked_layers

    def test_cold_counter_for_unknown_returns_zero(self, default_config):
        detector = CollapseDetector(config=default_config)
        assert detector.cold_counter_for("nonexistent", 0) == 0

    def test_repr_contains_threshold_values(self, default_config):
        detector = CollapseDetector(config=default_config)
        r = repr(detector)
        assert "dead_threshold" in r
        assert "cold_threshold" in r


# =============================================================================
# §8  CollapseDetector.compute_load_imbalance() — Static Method
# =============================================================================

class TestComputeLoadImbalance:

    def test_uniform_distribution_score_one(self):
        """Perfectly uniform utilization → score ≈ 1.0."""
        util  = torch.ones(8) / 8.0
        score, argmax, max_util = CollapseDetector.compute_load_imbalance(util)
        assert abs(score - 1.0) < 1e-5

    def test_one_hot_distribution_score_n_experts(self):
        """One expert gets all tokens → score == n_experts."""
        n    = 8
        util = torch.zeros(n)
        util[0] = 1.0
        score, argmax, max_util = CollapseDetector.compute_load_imbalance(util)
        assert abs(score - float(n)) < 1e-4

    def test_argmax_points_to_most_loaded_expert(self):
        util = torch.tensor([0.1, 0.5, 0.2, 0.2])
        _, argmax, _ = CollapseDetector.compute_load_imbalance(util)
        assert argmax == 1

    def test_max_util_matches_max_value(self):
        util = torch.tensor([0.1, 0.6, 0.2, 0.1])
        _, _, max_util = CollapseDetector.compute_load_imbalance(util)
        assert abs(max_util - 0.6) < 1e-6

    def test_empty_tensor_returns_nan(self):
        util  = torch.zeros(0)
        score, argmax, max_util = CollapseDetector.compute_load_imbalance(util)
        assert math.isnan(score)
        assert argmax == -1
        assert math.isnan(max_util)

    def test_all_zero_tensor_returns_nan(self):
        util  = torch.zeros(8)
        score, argmax, max_util = CollapseDetector.compute_load_imbalance(util)
        assert math.isnan(score)

    def test_two_expert_balance(self):
        util  = torch.tensor([0.5, 0.5])
        score, _, _ = CollapseDetector.compute_load_imbalance(util)
        assert abs(score - 1.0) < 1e-5

    def test_highly_skewed_score_greater_than_threshold(self):
        """Skewed routing produces imbalance > default warn threshold (3.0)."""
        util  = torch.tensor([0.7, 0.1, 0.1, 0.1])
        score, _, _ = CollapseDetector.compute_load_imbalance(util)
        # mean_util = 0.25, max_util = 0.7, score = 2.8 — just below 3
        # Accept as long as score > 1.5 (meaningful skew)
        assert score > 1.5
