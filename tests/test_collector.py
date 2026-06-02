# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  tests/test_collector.py — Unit tests for collector sub-package
#
#  COVERAGE TARGETS
#  ----------------
#    RingBuffer
#      ├── append() writes events in order
#      ├── capacity wrapping: oldest events are overwritten when full
#      ├── snapshot() returns oldest-first list
#      ├── latest(n) returns newest-first list
#      ├── len() reflects current occupancy
#      ├── is_full, has_wrapped, total_written properties
#      ├── clear() resets buffer without changing capacity
#      ├── TypeError on non-RoutingEvent append
#      ├── ValueError on capacity < 1
#      └── Thread-safety: snapshot() is lock-protected
#
#    StatCollector
#      ├── add_event() accumulates expert_counts correctly
#      ├── get_all_stats() returns LayerStats for every tracked layer
#      ├── get_layer_stats() returns None for unknown layers
#      ├── Rolling window: only recent window_steps events contribute
#      ├── Auto-registers unknown layers on first event
#      ├── Expert count sums equal total_tokens in LayerStats
#      ├── load_imbalance_score == 1.0 for perfectly balanced routing
#      ├── clear() resets a specific layer only
#      ├── clear() with no argument resets all layers
#      ├── tracked_layers and total_events properties
#      └── buffer_utilization() returns per-layer fill fractions
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#
# =============================================================================

from __future__ import annotations

import threading
import time
from typing import List

import pytest
import torch

from conftest import make_routing_event

from moewatch.collector.ring_buffer import RingBuffer
from moewatch.collector.stat_collector import StatCollector
from moewatch.hooks.router_hook import RoutingEvent
from moewatch.config import WatchConfig, OutputMode


# =============================================================================
# §1  RingBuffer
# =============================================================================

class TestRingBufferBasics:

    def test_initial_length_is_zero(self):
        buf = RingBuffer(capacity=10)
        assert len(buf) == 0

    def test_initial_snapshot_is_empty(self):
        buf = RingBuffer(capacity=10)
        assert buf.snapshot() == []

    def test_capacity_stored(self):
        buf = RingBuffer(capacity=42)
        assert buf.capacity == 42

    def test_append_increases_len(self):
        buf = RingBuffer(capacity=5)
        buf.append(make_routing_event(step=1))
        assert len(buf) == 1

    def test_append_three_events(self):
        buf = RingBuffer(capacity=10)
        for i in range(3):
            buf.append(make_routing_event(step=i))
        assert len(buf) == 3

    def test_snapshot_returns_oldest_first(self):
        buf = RingBuffer(capacity=5)
        for i in range(5):
            buf.append(make_routing_event(step=i))
        steps = [e.step for e in buf.snapshot()]
        assert steps == sorted(steps)

    def test_latest_returns_newest_first(self):
        buf = RingBuffer(capacity=10)
        for i in range(10):
            buf.append(make_routing_event(step=i))
        latest = buf.latest(3)
        assert len(latest) == 3
        assert latest[0].step > latest[1].step  # newest first

    def test_latest_clamped_to_len(self):
        buf = RingBuffer(capacity=10)
        buf.append(make_routing_event(step=1))
        assert len(buf.latest(100)) == 1

    def test_iter_same_as_snapshot(self):
        buf = RingBuffer(capacity=5)
        for i in range(5):
            buf.append(make_routing_event(step=i))
        assert list(buf) == buf.snapshot()


class TestRingBufferWrapping:

    def test_len_capped_at_capacity(self):
        buf = RingBuffer(capacity=3)
        for i in range(10):
            buf.append(make_routing_event(step=i))
        assert len(buf) == 3

    def test_is_full_after_capacity(self):
        buf = RingBuffer(capacity=3)
        for i in range(3):
            buf.append(make_routing_event(step=i))
        assert buf.is_full

    def test_not_full_before_capacity(self):
        buf = RingBuffer(capacity=5)
        for i in range(4):
            buf.append(make_routing_event(step=i))
        assert not buf.is_full

    def test_has_wrapped_after_overflow(self):
        buf = RingBuffer(capacity=3)
        for i in range(5):
            buf.append(make_routing_event(step=i))
        assert buf.has_wrapped

    def test_has_not_wrapped_when_not_full(self):
        buf = RingBuffer(capacity=10)
        for i in range(5):
            buf.append(make_routing_event(step=i))
        assert not buf.has_wrapped

    def test_oldest_events_overwritten_on_wrap(self):
        """When the buffer wraps, the oldest event is gone."""
        buf = RingBuffer(capacity=3)
        for i in range(5):
            buf.append(make_routing_event(step=i))
        steps = [e.step for e in buf.snapshot()]
        # Only the 3 most recent: steps 2, 3, 4
        assert 0 not in steps
        assert 4 in steps

    def test_snapshot_correct_after_wrap(self):
        """snapshot() returns exactly capacity items after wrap."""
        buf = RingBuffer(capacity=3)
        for i in range(7):
            buf.append(make_routing_event(step=i))
        snap = buf.snapshot()
        assert len(snap) == 3

    def test_total_written_exceeds_capacity_after_wrap(self):
        buf = RingBuffer(capacity=3)
        for i in range(10):
            buf.append(make_routing_event(step=i))
        assert buf.total_written == 10

    def test_utilization_is_one_when_full(self):
        buf = RingBuffer(capacity=4)
        for i in range(4):
            buf.append(make_routing_event(step=i))
        assert buf.utilization == 1.0


class TestRingBufferClearAndGuards:

    def test_clear_resets_len(self):
        buf = RingBuffer(capacity=5)
        for i in range(5):
            buf.append(make_routing_event(step=i))
        buf.clear()
        assert len(buf) == 0

    def test_clear_resets_snapshot(self):
        buf = RingBuffer(capacity=5)
        for i in range(5):
            buf.append(make_routing_event(step=i))
        buf.clear()
        assert buf.snapshot() == []

    def test_clear_does_not_change_capacity(self):
        buf = RingBuffer(capacity=7)
        for i in range(7):
            buf.append(make_routing_event(step=i))
        buf.clear()
        assert buf.capacity == 7

    def test_clear_resets_wrapped_flag(self):
        buf = RingBuffer(capacity=3)
        for i in range(5):
            buf.append(make_routing_event(step=i))
        assert buf.has_wrapped
        buf.clear()
        assert not buf.has_wrapped

    def test_append_non_routing_event_raises_type_error(self):
        buf = RingBuffer(capacity=5)
        with pytest.raises(TypeError, match="RoutingEvent"):
            buf.append("not a routing event")  # type: ignore[arg-type]

    def test_capacity_less_than_one_raises_value_error(self):
        with pytest.raises(ValueError):
            RingBuffer(capacity=0)

    def test_capacity_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            RingBuffer(capacity=-1)

    def test_capacity_non_int_raises_value_error(self):
        with pytest.raises((ValueError, TypeError)):
            RingBuffer(capacity=3.5)  # type: ignore[arg-type]


class TestRingBufferThreadSafety:

    def test_concurrent_snapshot_does_not_crash(self):
        """snapshot() can be called while writes happen on another thread."""
        buf  = RingBuffer(capacity=50)
        done = threading.Event()
        errors: List[Exception] = []

        def writer():
            for i in range(200):
                buf.append(make_routing_event(step=i))
                time.sleep(0)
            done.set()

        def reader():
            while not done.is_set():
                try:
                    _ = buf.snapshot()
                except Exception as exc:
                    errors.append(exc)

        t_write = threading.Thread(target=writer)
        t_read  = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join()
        t_read.join()

        assert errors == [], f"Thread-safety errors: {errors}"


# =============================================================================
# §2  StatCollector — Basic Accumulation
# =============================================================================

class TestStatCollectorBasics:

    def test_initial_tracked_layers(self, default_config):
        """tracked_layers reflects the layer names passed at construction."""
        names = ["layer0", "layer1"]
        coll  = StatCollector(layer_names=names, config=default_config)
        assert set(coll.tracked_layers) == set(names)

    def test_add_event_increments_total_events(self, default_config):
        coll = StatCollector(layer_names=["l0"], config=default_config)
        for i in range(5):
            coll.add_event(make_routing_event(layer_name="l0", step=i))
        assert coll.total_events == 5

    def test_add_event_accumulates_expert_counts(self, default_config):
        """Running expert counts equal the sum across all added events."""
        coll = StatCollector(layer_names=["l0"], config=default_config)
        per_event_counts = [10, 20, 30, 40, 50, 60, 70, 80]  # uniform 8 experts

        for i in range(3):
            coll.add_event(make_routing_event(
                layer_name="l0", step=i,
                expert_counts=per_event_counts,
            ))

        stats = coll.get_layer_stats("l0")
        assert stats is not None
        # Over 3 events, each expert gets 3× per_event_counts[i] tokens
        for i, expected in enumerate(per_event_counts):
            assert int(stats.expert_counts[i].item()) == expected * 3

    def test_get_all_stats_returns_all_layers(self, default_config):
        names = ["a", "b", "c"]
        coll  = StatCollector(layer_names=names, config=default_config)
        all_s = coll.get_all_stats()
        assert set(all_s.keys()) == set(names)

    def test_get_layer_stats_unknown_returns_none(self, default_config):
        coll = StatCollector(layer_names=["l0"], config=default_config)
        assert coll.get_layer_stats("non_existent") is None

    def test_empty_stats_when_no_events(self, default_config):
        coll  = StatCollector(layer_names=["l0"], config=default_config)
        stats = coll.get_layer_stats("l0")
        assert stats is not None
        assert stats.is_empty
        assert stats.event_count == 0

    def test_expert_counts_sum_equals_total_tokens(self, default_config):
        coll = StatCollector(layer_names=["l0"], config=default_config)
        for i in range(10):
            coll.add_event(make_routing_event(
                layer_name="l0", step=i,
                expert_counts=[100] * 8,
            ))
        stats = coll.get_layer_stats("l0")
        assert int(stats.expert_counts.sum().item()) == stats.total_tokens


class TestStatCollectorUtilization:

    def test_uniform_routing_load_imbalance_near_one(self, default_config):
        """Perfectly uniform routing → load_imbalance_score ≈ 1.0."""
        coll = StatCollector(layer_names=["l0"], config=default_config)
        for i in range(10):
            coll.add_event(make_routing_event(
                layer_name="l0", step=i,
                expert_counts=[100] * 8,
            ))
        stats = coll.get_layer_stats("l0")
        assert abs(stats.load_imbalance_score - 1.0) < 0.01

    def test_collapsed_routing_high_load_imbalance(self, default_config):
        """All tokens to one expert → load_imbalance_score == n_experts."""
        coll = StatCollector(layer_names=["l0"], config=default_config)
        for i in range(5):
            coll.add_event(make_routing_event(
                layer_name="l0", step=i,
                expert_counts=[800] + [0] * 7,
            ))
        stats = coll.get_layer_stats("l0")
        # max_util = 1.0, mean_util = 1/8, ratio = 8.0
        assert stats.load_imbalance_score > 4.0

    def test_utilization_sums_to_one_for_top1(self, default_config):
        """utilization tensor sums to 1.0 for top-1 routing."""
        coll = StatCollector(layer_names=["l0"], config=default_config)
        for i in range(5):
            coll.add_event(make_routing_event(
                layer_name="l0", step=i,
                expert_counts=[125] * 8,
            ))
        stats = coll.get_layer_stats("l0")
        total = stats.utilization.sum().item()
        assert abs(total - 1.0) < 1e-4

    def test_zero_tokens_layer_returns_nan_imbalance(self, default_config):
        """A layer with zero total tokens returns NaN load_imbalance_score."""
        coll  = StatCollector(layer_names=["l0"], config=default_config)
        stats = coll.get_layer_stats("l0")
        import math
        assert math.isnan(stats.load_imbalance_score)


class TestStatCollectorWindowBehaviour:

    def test_rolling_window_excludes_old_events(self):
        """With window_steps=3, only the 3 most-recent events contribute."""
        config = WatchConfig(
            output=OutputMode.SILENT,
            sample_every=1,
            window_steps=3,
            ring_buffer_capacity=100,
        )
        coll = StatCollector(layer_names=["l0"], config=config)

        # Add 10 events; only the last 3 should be in the window
        for i in range(10):
            coll.add_event(make_routing_event(
                layer_name="l0", step=i,
                expert_counts=[100] * 8,
            ))

        stats = coll.get_layer_stats("l0")
        # window_steps=3 → event_count should be ≤ 3
        assert stats.event_count <= 3


class TestStatCollectorClear:

    def test_clear_single_layer(self, default_config):
        """clear(layer_name) resets only the specified layer."""
        coll = StatCollector(layer_names=["l0", "l1"], config=default_config)
        for i in range(5):
            coll.add_event(make_routing_event(layer_name="l0", step=i))
            coll.add_event(make_routing_event(layer_name="l1", step=i))

        coll.clear(layer_name="l0")

        l0 = coll.get_layer_stats("l0")
        l1 = coll.get_layer_stats("l1")

        assert l0.event_count == 0    # cleared
        assert l1.event_count == 5    # untouched

    def test_clear_all_layers(self, default_config):
        """clear() with no argument clears all layers."""
        coll = StatCollector(layer_names=["l0", "l1"], config=default_config)
        for i in range(5):
            coll.add_event(make_routing_event(layer_name="l0", step=i))
            coll.add_event(make_routing_event(layer_name="l1", step=i))

        coll.clear()

        for name in ["l0", "l1"]:
            stats = coll.get_layer_stats(name)
            assert stats.event_count == 0

    def test_total_events_resets_after_clear(self, default_config):
        coll = StatCollector(layer_names=["l0"], config=default_config)
        for i in range(5):
            coll.add_event(make_routing_event(layer_name="l0", step=i))
        coll.clear()
        # ring_buffer.total_written tracks cumulative writes; event_count resets
        stats = coll.get_layer_stats("l0")
        assert stats.event_count == 0


class TestStatCollectorAutoRegister:

    def test_auto_registers_unknown_layer(self, default_config):
        """add_event() for an unknown layer auto-registers it."""
        coll = StatCollector(layer_names=["l0"], config=default_config)
        coll.add_event(make_routing_event(layer_name="new_layer", step=1))

        # new_layer should now be tracked
        assert "new_layer" in coll.tracked_layers

    def test_buffer_utilization_returns_fractions(self, default_config):
        """buffer_utilization() maps layer_name → float in [0, 1]."""
        coll = StatCollector(layer_names=["l0", "l1"], config=default_config)
        for i in range(5):
            coll.add_event(make_routing_event(layer_name="l0", step=i))

        util = coll.buffer_utilization()
        assert "l0" in util
        assert 0.0 <= util["l0"] <= 1.0
        assert util["l1"] == 0.0  # no events

    def test_summary_returns_string(self, seeded_collector):
        """summary() returns a non-empty string."""
        s = seeded_collector.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_repr_contains_layer_count(self, default_config):
        coll = StatCollector(layer_names=["a", "b"], config=default_config)
        r = repr(coll)
        assert "2" in r
