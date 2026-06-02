# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  tests/test_watcher.py — Unit tests for moewatch._watcher
#
#  COVERAGE TARGETS
#  ----------------
#    MoEWatch construction
#      ├── TypeError when model is not nn.Module
#      ├── Defaults to WatchConfig() when config=None
#      ├── is_attached is False before start()
#      └── alert_log is empty before start()
#
#    MoEWatch.start() / stop()
#      ├── start() sets is_attached = True
#      ├── stop() sets is_attached = False
#      ├── stop() cleans up all forward hooks
#      ├── duplicate start() emits UserWarning, no double hooks
#      ├── stop() is safe to call multiple times (idempotent)
#      ├── RuntimeError when no router modules detected and none specified
#      └── start() returns self for chaining
#
#    MoEWatch context manager
#      ├── __enter__ / __exit__ attach and detach correctly
#      ├── hooks detached even when exception propagates
#      └── exception is NOT suppressed by __exit__
#
#    MoEWatch.step()
#      ├── Returns [] when not attached
#      ├── Returns [] when step % log_every != 0
#      ├── Returns a list of Alert objects at log_every steps
#      ├── INFO heartbeat returned when all experts healthy
#      ├── WARN alert returned for cold experts
#      ├── ERROR alert returned for dead experts and low entropy
#      ├── Alert has step, level, layer_name, message attributes
#      ├── get_alert_log() accumulates all alerts across steps
#      ├── get_alert_log_json() returns valid JSON
#      └── summary() reflects total alert counts
#
#    Alert data class
#      ├── to_dict() returns required keys
#      ├── to_json_line() returns valid JSON string
#      └── timestamp is set to a recent wall-clock time
#
#    MoEWatch.attach(trainer)
#      ├── Registers MoEWatchCallback with the trainer
#      ├── TypeError when trainer has no add_callback attribute
#      └── is_attached True after attach()
#
#    MoEWatchCallback
#      ├── on_log() calls watcher.step(global_step)
#      └── on_train_end() calls watcher.stop()
#
#    Additional branch coverage
#      ├── _colour(no_color=True)
#      ├── step() returns [] when collector has no stats
#      ├── Low-entropy WARN alert path
#      ├── Cold-expert WARN alert path
#      ├── JSON alert emission path
#      ├── Console alert emission path
#      ├── ERROR suggestion rendering path
#      ├── Startup banner long-name truncation
#      ├── Startup banner "... and N more layer(s)" path
#      ├── Startup banner suppressed in SILENT mode
#      └── TrainerCallback import fallback path
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#
# =============================================================================

from __future__ import annotations

import json
import time
import warnings
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from conftest import SyntheticMoE, run_forward_passes

from moewatch import MoEWatch, WatchConfig
from moewatch._watcher import Alert, MoEWatchCallback
from moewatch.config import AlertLevel, OutputMode


# =============================================================================
# §1  Helpers
# =============================================================================

def _make_watcher(model: nn.Module, **config_kwargs) -> MoEWatch:
    """Build a MoEWatch with SILENT output and sample_every=1 for tests."""
    config = WatchConfig(
        router_modules=getattr(model, "router_module_names", []),
        output=OutputMode.SILENT,
        sample_every=1,
        log_every=1,
        **config_kwargs,
    )
    return MoEWatch(model, config=config)


def _make_trainer_mock(watcher: MoEWatch) -> SimpleNamespace:
    """Minimal duck-type HuggingFace Trainer mock."""
    callbacks: List = []

    def add_callback(cb):
        callbacks.append(cb)

    trainer = SimpleNamespace(add_callback=add_callback, _callbacks=callbacks)
    return trainer


# =============================================================================
# §2  MoEWatch Construction
# =============================================================================

class TestMoEWatchConstruction:

    def test_type_error_for_non_module(self):
        with pytest.raises(TypeError, match="nn.Module"):
            MoEWatch("not a model")

    def test_default_config_when_none(self, synthetic_model):
        watcher = MoEWatch(synthetic_model, config=None)
        assert isinstance(watcher.config, WatchConfig)

    def test_is_attached_false_before_start(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        assert watcher.is_attached is False

    def test_alert_log_empty_before_start(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        assert watcher.get_alert_log() == []

    def test_global_step_zero_before_start(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        assert watcher.global_step == 0

    def test_repr_contains_model_name(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        assert "SyntheticMoE" in repr(watcher)

    def test_repr_shows_detached_before_start(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        assert "detached" in repr(watcher)


# =============================================================================
# §3  MoEWatch.start() / stop()
# =============================================================================

class TestMoEWatchStartStop:

    def test_start_sets_is_attached(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            assert watcher.is_attached is True
        finally:
            watcher.stop()

    def test_stop_sets_is_attached_false(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        watcher.stop()
        assert watcher.is_attached is False

    def test_stop_removes_all_hooks(self, synthetic_model):
        """No forward hooks remain on any submodule after stop()."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        watcher.stop()

        for module in synthetic_model.modules():
            assert len(module._forward_hooks) == 0

    def test_stop_idempotent(self, synthetic_model):
        """Calling stop() multiple times does not raise."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        watcher.stop()
        watcher.stop()   # second call — should be a no-op
        watcher.stop()   # third call — still fine

    def test_start_returns_self(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        result  = watcher.start()
        try:
            assert result is watcher
        finally:
            watcher.stop()

    def test_stop_returns_self(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        result = watcher.stop()
        assert result is watcher

    def test_duplicate_start_emits_user_warning(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                watcher.start()
            assert any(issubclass(x.category, UserWarning) for x in w)
        finally:
            watcher.stop()

    def test_duplicate_start_does_not_double_attach_hooks(self, synthetic_model):
        """A duplicate start() call must not register extra hooks."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                watcher.start()

            for module in synthetic_model.modules():
                assert len(module._forward_hooks) <= 1
        finally:
            watcher.stop()

    def test_no_router_modules_raises_runtime_error(self):
        """start() raises RuntimeError when no router modules are detected and none specified."""
        class _Dense(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        dense   = _Dense()
        config  = WatchConfig(output=OutputMode.SILENT, sample_every=1)
        watcher = MoEWatch(dense, config=config)

        with pytest.raises(RuntimeError, match="router"):
            watcher.start()

    def test_start_stop_multiple_cycles(self, synthetic_model):
        """start→stop can be repeated without leaking hooks."""
        for _ in range(3):
            watcher = _make_watcher(synthetic_model)
            watcher.start()
            assert watcher.is_attached
            watcher.stop()
            assert not watcher.is_attached

        for module in synthetic_model.modules():
            assert len(module._forward_hooks) == 0


# =============================================================================
# §4  MoEWatch Context Manager
# =============================================================================

class TestMoEWatchContextManager:

    def test_enter_attaches_hooks(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        with watcher:
            assert watcher.is_attached is True

    def test_exit_detaches_hooks(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        with watcher:
            pass
        assert watcher.is_attached is False

    def test_exit_detaches_on_exception(self, synthetic_model):
        """Hooks are removed even when an exception propagates."""
        watcher = _make_watcher(synthetic_model)
        with pytest.raises(RuntimeError, match="intentional"):
            with watcher:
                raise RuntimeError("intentional")
        assert watcher.is_attached is False

    def test_exception_not_suppressed(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        raised  = False
        try:
            with watcher:
                raise ValueError("should propagate")
        except ValueError:
            raised = True
        assert raised

    def test_hooks_fire_inside_context(self, synthetic_model):
        """Forward passes inside the with-block write routing events."""
        watcher = _make_watcher(synthetic_model)
        with watcher:
            run_forward_passes(synthetic_model, n_passes=3)
            alerts = watcher.step(1)
        # At least a heartbeat should be returned at step 1 (log_every=1)
        assert isinstance(alerts, list)


# =============================================================================
# §5  MoEWatch.step() — Alert Emission
# =============================================================================

class TestMoEWatchStep:

    def test_step_returns_empty_list_when_not_attached(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        result  = watcher.step(100)
        assert result == []

    def test_step_returns_list_of_alerts(self, synthetic_model):
        """step() at a log_every boundary returns a list of Alert objects."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            run_forward_passes(synthetic_model, n_passes=5)
            alerts = watcher.step(1)   # log_every=1 so this triggers
            assert isinstance(alerts, list)
        finally:
            watcher.stop()

    def test_step_skipped_on_non_boundary(self, synthetic_model):
        """step() returns [] on steps not divisible by log_every."""
        config  = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
            log_every=10,   # only fire every 10 steps
        )
        watcher = MoEWatch(synthetic_model, config=config)
        watcher.start()
        try:
            run_forward_passes(synthetic_model, n_passes=3)
            result = watcher.step(7)   # 7 % 10 != 0 → skip
            assert result == []
        finally:
            watcher.stop()

    def test_step_fires_on_log_every_boundary(self, synthetic_model):
        """step() returns non-empty list at a log_every boundary."""
        config  = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
            log_every=5,
        )
        watcher = MoEWatch(synthetic_model, config=config)
        watcher.start()
        try:
            run_forward_passes(synthetic_model, n_passes=10)
            alerts = watcher.step(5)   # 5 % 5 == 0
            assert len(alerts) > 0
        finally:
            watcher.stop()

    def test_healthy_model_info_heartbeat(self, synthetic_model):
        """Healthy model produces an INFO heartbeat at each log step."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            run_forward_passes(synthetic_model, n_passes=5)
            alerts = watcher.step(1)
            info_alerts = [a for a in alerts if a.level == AlertLevel.INFO]
            assert len(info_alerts) >= 1
        finally:
            watcher.stop()

    def test_collapsed_model_produces_error_alert(self, tiny_dataloader):
        """A collapsed model produces at least one ERROR-level alert."""
        model  = SyntheticMoE(n_layers=2, n_experts=8, hidden_dim=64, collapse_layer=0)
        config = WatchConfig(
            router_modules=model.router_module_names,
            dead_threshold=0.001,
            cold_threshold=0.005,
            output=OutputMode.SILENT,
            sample_every=1,
            log_every=1,
        )
        watcher = MoEWatch(model, config=config)
        watcher.start()
        try:
            run_forward_passes(model, n_passes=20)
            alerts = watcher.step(1)
            error_alerts = [a for a in alerts if a.level == AlertLevel.ERROR]
            assert len(error_alerts) >= 1
        finally:
            watcher.stop()

    def test_alert_has_required_attributes(self, synthetic_model):
        """Every Alert returned by step() has the documented attributes."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            run_forward_passes(synthetic_model, n_passes=3)
            alerts = watcher.step(1)
            for alert in alerts:
                assert hasattr(alert, "step")
                assert hasattr(alert, "level")
                assert hasattr(alert, "layer_name")
                assert hasattr(alert, "message")
                assert isinstance(alert.message, str)
        finally:
            watcher.stop()

    def test_get_alert_log_accumulates(self, synthetic_model):
        """get_alert_log() accumulates alerts across multiple step() calls."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            for step in range(1, 4):
                run_forward_passes(synthetic_model, n_passes=2)
                watcher.step(step)

            log = watcher.get_alert_log()
            assert len(log) >= 3    # at least one alert per step call
        finally:
            watcher.stop()

    def test_get_alert_log_returns_copy(self, synthetic_model):
        """get_alert_log() returns a copy — mutating it doesn't affect the watcher."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            run_forward_passes(synthetic_model, n_passes=2)
            watcher.step(1)
            log = watcher.get_alert_log()
            original_len = len(log)
            log.clear()
            assert len(watcher.get_alert_log()) == original_len
        finally:
            watcher.stop()

    def test_get_alert_log_json_is_valid_json(self, synthetic_model):
        """get_alert_log_json() produces a parseable JSON array."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            run_forward_passes(synthetic_model, n_passes=2)
            watcher.step(1)
            json_str = watcher.get_alert_log_json()
            parsed   = json.loads(json_str)
            assert isinstance(parsed, list)
        finally:
            watcher.stop()

    def test_summary_returns_string(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            run_forward_passes(synthetic_model, n_passes=2)
            watcher.step(1)
            s = watcher.summary()
            assert isinstance(s, str)
            assert "alert" in s.lower()
        finally:
            watcher.stop()


# =============================================================================
# §6  Alert Data Class
# =============================================================================

class TestAlertDataClass:

    def _make_alert(self, level: AlertLevel = AlertLevel.INFO) -> Alert:
        return Alert(
            step=42,
            level=level,
            layer_name="moe_layers.0",
            message="Test alert message.",
            metric=0.003,
            suggestion="Raise aux_loss_coef." if level == AlertLevel.ERROR else None,
        )

    def test_to_dict_required_keys(self):
        alert = self._make_alert()
        d     = alert.to_dict()
        for key in ("step", "level", "layer_name", "message", "metric",
                    "suggestion", "timestamp"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_level_is_string(self):
        alert = self._make_alert(AlertLevel.ERROR)
        d     = alert.to_dict()
        assert isinstance(d["level"], str)
        assert d["level"] == "ERROR"

    def test_to_json_line_is_valid_json(self):
        alert = self._make_alert()
        line  = alert.to_json_line()
        parsed = json.loads(line)
        assert parsed["step"] == 42

    def test_timestamp_recent(self):
        before = time.time()
        alert  = self._make_alert()
        after  = time.time()
        assert before <= alert.timestamp <= after

    def test_info_alert_no_suggestion(self):
        """INFO alerts typically have no suggestion."""
        alert = self._make_alert(AlertLevel.INFO)
        # suggestion is None for INFO
        assert alert.suggestion is None

    def test_error_alert_has_suggestion(self):
        alert = self._make_alert(AlertLevel.ERROR)
        assert alert.suggestion is not None


# =============================================================================
# §7  MoEWatch.attach(trainer) — HuggingFace integration
# =============================================================================

class TestMoEWatchAttachTrainer:

    def test_attach_sets_is_attached(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        trainer = _make_trainer_mock(watcher)
        watcher.attach(trainer)
        try:
            assert watcher.is_attached is True
        finally:
            watcher.stop()

    def test_attach_registers_callback(self, synthetic_model):
        """attach() calls trainer.add_callback exactly once."""
        watcher  = _make_watcher(synthetic_model)
        trainer  = _make_trainer_mock(watcher)
        watcher.attach(trainer)
        try:
            assert len(trainer._callbacks) == 1
            assert isinstance(trainer._callbacks[0], MoEWatchCallback)
        finally:
            watcher.stop()

    def test_attach_type_error_no_add_callback(self, synthetic_model):
        """attach() raises TypeError if trainer has no add_callback method."""
        watcher = _make_watcher(synthetic_model)
        fake_trainer = SimpleNamespace()   # no add_callback

        with pytest.raises(TypeError, match="add_callback"):
            watcher.attach(fake_trainer)

    def test_detach_alias_for_stop(self, synthetic_model):
        """detach() is an alias for stop()."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        watcher.detach()   # should behave identically to stop()
        assert watcher.is_attached is False


# =============================================================================
# §8  MoEWatchCallback
# =============================================================================

class TestMoEWatchCallback:

    def _make_state(self, global_step: int = 100) -> SimpleNamespace:
        return SimpleNamespace(global_step=global_step)

    def test_on_log_calls_watcher_step(self, synthetic_model):
        """on_log() forwards global_step to watcher.step()."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            step_calls = []

            original_step = watcher.step

            def mock_step(s):
                step_calls.append(s)
                return original_step(s)

            watcher.step = mock_step
            cb = MoEWatchCallback(watcher=watcher)

            state = self._make_state(global_step=50)
            cb.on_log(args=None, state=state, control=None)

            assert 50 in step_calls
        finally:
            watcher.stop()

    def test_on_train_end_calls_stop(self, synthetic_model):
        """on_train_end() detaches hooks via watcher.stop()."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        assert watcher.is_attached

        cb = MoEWatchCallback(watcher=watcher)
        cb.on_train_end(args=None, state=self._make_state(), control=None)

        assert watcher.is_attached is False

    def test_on_train_end_idempotent_when_already_stopped(self, synthetic_model):
        """on_train_end() is safe to call when the watcher is already stopped."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        watcher.stop()  # already stopped

        cb = MoEWatchCallback(watcher=watcher)
        # Should not raise
        cb.on_train_end(args=None, state=self._make_state(), control=None)
        assert watcher.is_attached is False

    def test_repr_contains_watcher(self, synthetic_model):
        watcher = _make_watcher(synthetic_model)
        cb      = MoEWatchCallback(watcher=watcher)
        assert "MoEWatch" in repr(cb)

    def test_on_log_state_without_global_step_defaults_to_zero(self, synthetic_model):
        """on_log() handles state objects with no global_step attribute gracefully."""
        watcher = _make_watcher(synthetic_model)
        watcher.start()
        try:
            cb    = MoEWatchCallback(watcher=watcher)
            state = SimpleNamespace()   # no global_step
            run_forward_passes(synthetic_model, n_passes=2)
            # Should not raise
            cb.on_log(args=None, state=state, control=None)
        finally:
            watcher.stop()

# =============================================================================
# Extra coverage tests for uncovered branches in _watcher.py
# =============================================================================

from types import SimpleNamespace
from unittest.mock import MagicMock

from moewatch._watcher import Alert
from moewatch.config import AlertLevel, OutputMode


def test_warn_entropy_alert_branch(synthetic_model):
    watcher = _make_watcher(synthetic_model)
    watcher.start()

    try:
        watcher._collector.get_all_stats = MagicMock(
            return_value={
                "router": SimpleNamespace(load_imbalance_score=1.0)
            }
        )

        watcher._entropy_analyzer.analyze = MagicMock(
            return_value=SimpleNamespace(
                results={
                    "router": SimpleNamespace(
                        alert_level=AlertLevel.WARN,
                        entropy_norm=0.20,
                    )
                }
            )
        )

        watcher._collapse_detector.detect = MagicMock(return_value={})

        alerts = watcher.step(1)

        assert any(
            "Routing entropy LOW" in a.message
            for a in alerts
        )

    finally:
        watcher.stop()


def test_warn_cold_expert_alert_branch(synthetic_model):
    watcher = _make_watcher(synthetic_model)
    watcher.start()

    try:
        watcher._collector.get_all_stats = MagicMock(
            return_value={
                "router": SimpleNamespace(load_imbalance_score=1.0)
            }
        )

        watcher._entropy_analyzer.analyze = MagicMock(
            return_value=SimpleNamespace(results={})
        )

        cold_expert = SimpleNamespace(
            is_dead=False,
            is_cold=True,
            expert_idx=3,
            utilization=0.003,
        )

        watcher._collapse_detector.detect = MagicMock(
            return_value={
                "router": SimpleNamespace(
                    experts=[cold_expert]
                )
            }
        )

        alerts = watcher.step(1)

        assert any(
            "COLD" in a.message
            for a in alerts
        )

    finally:
        watcher.stop()


def test_emit_alerts_console_error_suggestion(capsys, synthetic_model):
    config = WatchConfig(
        router_modules=synthetic_model.router_module_names,
        output=OutputMode.CONSOLE,
        sample_every=1,
        log_every=1,
    )

    watcher = MoEWatch(synthetic_model, config)

    alert = Alert(
        step=1,
        level=AlertLevel.ERROR,
        layer_name="router",
        message="dead expert",
        suggestion="increase aux loss",
    )

    watcher._emit_alerts([alert])

    out = capsys.readouterr().out

    assert "Suggestion" in out
    assert "increase aux loss" in out


def test_print_banner_many_layers(capsys, synthetic_model):
    config = WatchConfig(
        router_modules=synthetic_model.router_module_names,
        output=OutputMode.CONSOLE,
    )

    watcher = MoEWatch(synthetic_model, config)

    names = [f"layer_{i}" for i in range(10)]

    watcher._print_banner(names)

    out = capsys.readouterr().out

    assert "... and 5 more layer(s)" in out


def test_print_banner_long_layer_name(capsys, synthetic_model):
    config = WatchConfig(
        router_modules=synthetic_model.router_module_names,
        output=OutputMode.CONSOLE,
    )

    watcher = MoEWatch(synthetic_model, config)

    long_name = "x" * 100

    watcher._print_banner([long_name])

    out = capsys.readouterr().out

    assert "x" * 54 in out
