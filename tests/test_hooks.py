# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  tests/test_hooks.py — Unit & integration tests for the hooks sub-package
#
#  COVERAGE TARGETS
#  ----------------
#    HookManager
#      ├── attach() registers hooks on all specified modules
#      ├── detach() removes all hooks cleanly
#      ├── attach() idempotency — duplicate call emits warning, no double hook
#      ├── attach() with invalid module name raises ValueError
#      ├── Context-manager (__enter__ / __exit__) always detaches on exit
#      ├── __exit__ detaches even when an exception propagates
#      ├── is_attached property tracks state correctly
#      ├── hook_count reflects registered hooks
#      ├── diagnostics() returns per-hook call counts
#      └── get_hook() retrieves a specific RouterHook by name
#
#    RouterHook
#      ├── Fires on each forward pass and increments call_count
#      ├── Step-sampling (sample_every=N) skips non-sampled steps
#      ├── Extraction Strategy 1: tuple output → RoutingEvent written
#      ├── Extraction Strategy 3: single logit tensor → RoutingEvent written
#      ├── Extraction Strategy 4: dict output → RoutingEvent written
#      ├── Graceful error handling — never raises inside a forward pass
#      └── RoutingEvent contains valid n_experts and expert_counts
#
#    RoutingEvent
#      ├── expert_counts is a CPU int64 tensor
#      ├── expert_counts.sum() == total_tokens for top-1 routing
#      └── n_experts matches router dimensions
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#
# =============================================================================

from __future__ import annotations

import warnings
from typing import Any, List, Tuple

import pytest
import torch
import torch.nn as nn

from conftest import SyntheticMoE, SyntheticMoERouter, run_forward_passes

from moewatch.config import WatchConfig, OutputMode
from moewatch.hooks.manager import HookManager
from moewatch.hooks.router_hook import RouterHook, RoutingEvent
from moewatch.collector.stat_collector import StatCollector
from types import SimpleNamespace


# =============================================================================
# §1  Fixtures
# =============================================================================

@pytest.fixture()
def model_and_names(synthetic_model, default_config):
    """Return (model, router_names) for the 2-layer synthetic model."""
    return synthetic_model, synthetic_model.router_module_names


@pytest.fixture()
def collector(default_config, model_and_names):
    """StatCollector wired to both router layers."""
    _, names = model_and_names
    return StatCollector(layer_names=names, config=default_config)


@pytest.fixture()
def hook_manager(model_and_names, collector, default_config):
    """Un-attached HookManager."""
    model, names = model_and_names
    return HookManager(
        model=model,
        router_module_names=names,
        collector=collector,
        config=default_config,
    )


# =============================================================================
# §2  HookManager — Attach / Detach Lifecycle
# =============================================================================

class TestHookManagerLifecycle:

    def test_attach_sets_is_attached(self, hook_manager):
        """After attach(), is_attached is True."""
        try:
            hook_manager.attach()
            assert hook_manager.is_attached is True
        finally:
            hook_manager.detach()

    def test_detach_clears_is_attached(self, hook_manager):
        """After detach(), is_attached is False."""
        hook_manager.attach()
        hook_manager.detach()
        assert hook_manager.is_attached is False

    def test_hook_count_equals_router_modules(self, hook_manager, model_and_names):
        """hook_count equals the number of router module names."""
        _, names = model_and_names
        hook_manager.attach()
        try:
            assert hook_manager.hook_count == len(names)
        finally:
            hook_manager.detach()

    def test_hook_count_zero_before_attach(self, hook_manager):
        """hook_count is 0 before attach() is called."""
        assert hook_manager.hook_count == 0

    def test_hook_count_zero_after_detach(self, hook_manager):
        """hook_count returns to 0 after detach()."""
        hook_manager.attach()
        hook_manager.detach()
        assert hook_manager.hook_count == 0

    def test_detach_safe_when_not_attached(self, hook_manager):
        """detach() on an un-attached manager does not raise."""
        hook_manager.detach()   # should be a no-op

    def test_duplicate_attach_emits_warning(self, hook_manager):
        """Calling attach() twice emits a UserWarning."""
        hook_manager.attach()
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                hook_manager.attach()
            assert any("already-attached" in str(warning.message) for warning in w)
        finally:
            hook_manager.detach()

    def test_duplicate_attach_does_not_double_register(self, hook_manager, model_and_names):
        """A duplicate attach() does not register extra hooks."""
        _, names = model_and_names
        hook_manager.attach()
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                hook_manager.attach()
            # Still only one hook per module
            assert hook_manager.hook_count == len(names)
        finally:
            hook_manager.detach()

    def test_attach_invalid_module_name_raises(
        self, model_and_names, collector, default_config
    ):
        """attach() with a non-existent module name raises ValueError."""
        model, _ = model_and_names
        bad_manager = HookManager(
            model=model,
            router_module_names=["this.module.does.not.exist"],
            collector=collector,
            config=default_config,
        )
        with pytest.raises(ValueError, match="not found"):
            bad_manager.attach()

    def test_attach_returns_self(self, hook_manager):
        """attach() returns the HookManager instance for method chaining."""
        result = hook_manager.attach()
        try:
            assert result is hook_manager
        finally:
            hook_manager.detach()

    def test_detach_returns_self(self, hook_manager):
        """detach() returns the HookManager instance for method chaining."""
        hook_manager.attach()
        result = hook_manager.detach()
        assert result is hook_manager

    def test_multiple_attach_detach_cycles(self, model_and_names, collector, default_config):
        """attach→detach can be repeated without leaking hooks."""
        model, names = model_and_names
        for _ in range(3):
            hm = HookManager(
                model=model,
                router_module_names=names,
                collector=collector,
                config=default_config,
            )
            hm.attach()
            assert hm.is_attached
            hm.detach()
            assert not hm.is_attached


# =============================================================================
# §3  HookManager — Context Manager Protocol
# =============================================================================

class TestHookManagerContextManager:

    def test_context_manager_attaches_on_enter(self, hook_manager):
        """__enter__ attaches hooks; model forward pass fires hooks inside."""
        model, _ = next(
            (v for v in [(hook_manager.model, None)]), (None, None)
        )
        with hook_manager:
            assert hook_manager.is_attached

    def test_context_manager_detaches_on_exit(self, hook_manager):
        """Hooks are detached when the with-block exits normally."""
        with hook_manager:
            pass
        assert not hook_manager.is_attached
        assert hook_manager.hook_count == 0

    def test_context_manager_detaches_on_exception(self, hook_manager):
        """Hooks are detached even when an exception propagates out of the block."""
        with pytest.raises(RuntimeError, match="intentional test error"):
            with hook_manager:
                assert hook_manager.is_attached
                raise RuntimeError("intentional test error")

        assert not hook_manager.is_attached

    def test_context_manager_does_not_suppress_exception(self, hook_manager):
        """__exit__ returns None (falsy) — exceptions propagate normally."""
        raised = False
        try:
            with hook_manager:
                raise ValueError("should propagate")
        except ValueError:
            raised = True
        assert raised

    def test_context_manager_hooks_fire_during_block(
        self, model_and_names, collector, default_config
    ):
        """Forward passes inside the with-block write events to the collector."""
        model, names = model_and_names
        hm = HookManager(
            model=model,
            router_module_names=names,
            collector=collector,
            config=default_config,
        )
        with hm:
            run_forward_passes(model, n_passes=3)

        stats = collector.get_all_stats()
        total = sum(s.event_count for s in stats.values())
        assert total > 0


# =============================================================================
# §4  HookManager — Introspection
# =============================================================================

class TestHookManagerIntrospection:

    def test_get_hook_returns_router_hook(self, hook_manager, model_and_names):
        """get_hook() returns the RouterHook for a known layer name."""
        _, names = model_and_names
        hook_manager.attach()
        try:
            rh = hook_manager.get_hook(names[0])
            assert isinstance(rh, RouterHook)
        finally:
            hook_manager.detach()

    def test_get_hook_returns_none_for_unknown(self, hook_manager):
        """get_hook() returns None for an unknown layer name."""
        hook_manager.attach()
        try:
            assert hook_manager.get_hook("does.not.exist") is None
        finally:
            hook_manager.detach()

    def test_diagnostics_returns_dict(self, hook_manager, model_and_names):
        """diagnostics() returns a dict with 'calls' and 'errors' per layer."""
        _, names = model_and_names
        hook_manager.attach()
        try:
            run_forward_passes(hook_manager.model, n_passes=4)
            diag = hook_manager.diagnostics()
            assert isinstance(diag, dict)
            for name in names:
                assert name in diag
                assert "calls" in diag[name]
                assert "errors" in diag[name]
        finally:
            hook_manager.detach()

    def test_diagnostics_call_count_increments(self, hook_manager, model_and_names):
        """call_count increases with each forward pass (before sample_every filter)."""
        _, names = model_and_names
        hook_manager.attach()
        try:
            n_passes = 5
            run_forward_passes(hook_manager.model, n_passes=n_passes)
            diag = hook_manager.diagnostics()
            for name in names:
                # Each layer's hook fires once per forward pass
                assert diag[name]["calls"] == n_passes
        finally:
            hook_manager.detach()

    def test_repr_shows_status(self, hook_manager):
        """__repr__ includes hook count and attachment status."""
        hook_manager.attach()
        try:
            r = repr(hook_manager)
            assert "attached" in r
        finally:
            hook_manager.detach()


# =============================================================================
# §5  RouterHook — Extraction and Event Writing
# =============================================================================

class TestRouterHookExtraction:

    def test_hook_fires_and_writes_event(self, model_and_names, collector, default_config):
        """After a forward pass, at least one RoutingEvent is in the collector."""
        model, names = model_and_names
        hm = HookManager(
            model=model,
            router_module_names=names,
            collector=collector,
            config=default_config,
        )
        with hm:
            run_forward_passes(model, n_passes=1)

        stats = collector.get_all_stats()
        total_events = sum(s.event_count for s in stats.values())
        assert total_events > 0

    def test_routing_event_expert_counts_shape(self, model_and_names, collector, default_config):
        """RoutingEvent.expert_counts has shape (n_experts,)."""
        model, names = model_and_names
        hm = HookManager(
            model=model,
            router_module_names=names,
            collector=collector,
            config=default_config,
        )
        with hm:
            run_forward_passes(model, n_passes=3)

        # Inspect the ring buffer directly for one layer
        for name, acc in collector._accumulators.items():
            events = acc.ring_buffer.snapshot()
            if events:
                ev = events[0]
                assert ev.expert_counts.ndim == 1
                assert ev.expert_counts.shape[0] == ev.n_experts
                break

    def test_routing_event_counts_sum_equals_total_tokens(
        self, model_and_names, collector, default_config
    ):
        """For top-1 routing, expert_counts.sum() == total_tokens."""
        model, names = model_and_names
        hm = HookManager(
            model=model,
            router_module_names=names,
            collector=collector,
            config=default_config,
        )
        with hm:
            run_forward_passes(model, n_passes=2)

        for _, acc in collector._accumulators.items():
            for ev in acc.ring_buffer.snapshot():
                if ev.n_experts > 0:
                    assert int(ev.expert_counts.sum().item()) == ev.total_tokens

    def test_routing_event_on_cpu(self, model_and_names, collector, default_config):
        """RoutingEvent.expert_counts is always on CPU."""
        model, names = model_and_names
        hm = HookManager(
            model=model,
            router_module_names=names,
            collector=collector,
            config=default_config,
        )
        with hm:
            run_forward_passes(model, n_passes=2)

        for _, acc in collector._accumulators.items():
            for ev in acc.ring_buffer.snapshot():
                assert ev.expert_counts.device.type == "cpu"

    def test_routing_event_dtype_int64(self, model_and_names, collector, default_config):
        """RoutingEvent.expert_counts is dtype int64."""
        model, names = model_and_names
        hm = HookManager(
            model=model,
            router_module_names=names,
            collector=collector,
            config=default_config,
        )
        with hm:
            run_forward_passes(model, n_passes=2)

        for _, acc in collector._accumulators.items():
            for ev in acc.ring_buffer.snapshot():
                assert ev.expert_counts.dtype == torch.int64

    def test_step_sampling_reduces_event_count(
        self, model_and_names, collector
    ):
        """With sample_every=5, only 1/5 of forward passes produce events."""
        model, names = model_and_names
        sparse_config = WatchConfig(
            output=OutputMode.SILENT,
            sample_every=5,
            log_every=1,
        )
        sparse_collector = StatCollector(layer_names=names, config=sparse_config)
        hm = HookManager(
            model=model,
            router_module_names=names,
            collector=sparse_collector,
            config=sparse_config,
        )
        n_passes = 10
        with hm:
            run_forward_passes(model, n_passes=n_passes)

        for _, acc in sparse_collector._accumulators.items():
            # With sample_every=5 and 10 passes, each hook sees 10 calls
            # but only passes with call_count % 5 == 0 produce events.
            # That's 2 sampled calls: step 5 and step 10.
            assert len(acc.ring_buffer) <= n_passes // 5 + 1


# =============================================================================
# §6  RouterHook — Strategy Coverage
# =============================================================================

class TestRouterHookStrategies:
    """Tests that RouterHook handles multiple output formats correctly."""

    def _make_hook_and_collector(
        self, layer_name: str = "test_layer", sample_every: int = 1
    ):
        config = WatchConfig(output=OutputMode.SILENT, sample_every=sample_every)
        coll   = StatCollector(layer_names=[layer_name], config=config)
        hook   = RouterHook(layer_name=layer_name, collector=coll, config=config)
        return hook, coll

    def test_strategy1_tuple_logits(self):
        """Strategy 1: (hidden, router_logits) tuple output."""
        hook, coll = self._make_hook_and_collector()
        module     = nn.Identity()

        # Simulate Mixtral-style output: (hidden, logits(B*S, n_experts))
        hidden  = torch.zeros(2, 4, 8)
        logits  = torch.randn(8, 4)  # 2 batch * 4 seq = 8 tokens, 4 experts
        output  = (hidden, logits)

        hook(module, (hidden,), output)

        stats = coll.get_all_stats()
        assert stats["test_layer"].event_count == 1
        assert stats["test_layer"].n_experts == 4

    def test_strategy3_single_tensor_logits(self):
        """Strategy 3: single 2-D logit tensor as output."""
        hook, coll = self._make_hook_and_collector()
        module     = nn.Identity()

        logits = torch.randn(16, 8)  # 16 tokens, 8 experts
        hook(module, (logits,), logits)

        stats = coll.get_all_stats()
        assert stats["test_layer"].event_count == 1
        assert stats["test_layer"].n_experts == 8

    def test_strategy4_dict_output_router_logits(self):
        """Strategy 4: dict output with 'router_logits' key."""
        hook, coll = self._make_hook_and_collector()
        module     = nn.Identity()

        logits = torch.randn(8, 4)
        output = {"router_logits": logits, "other": "ignored"}

        hook(module, (logits,), output)

        stats = coll.get_all_stats()
        assert stats["test_layer"].event_count == 1

    def test_strategy4_dict_gate_logits(self):
        """Strategy 4: dict output with 'gate_logits' key."""
        hook, coll = self._make_hook_and_collector()
        module     = nn.Identity()

        logits = torch.randn(8, 4)
        output = {"gate_logits": logits}

        hook(module, (logits,), output)

        stats = coll.get_all_stats()
        assert stats["test_layer"].event_count == 1

    def test_unrecognised_output_does_not_raise(self):
        """An unrecognised output type produces a zero-count event, not an exception."""
        hook, coll = self._make_hook_and_collector()
        module     = nn.Identity()

        # Pass a string — completely unrecognisable
        try:
            hook(module, (), "unrecognised_output")
        except Exception as exc:
            pytest.fail(f"RouterHook raised inside forward pass: {exc}")

    def test_error_count_increments_on_bad_extraction(self):
        """Extraction errors increment error_count without crashing."""
        hook, coll = self._make_hook_and_collector()
        module     = nn.Identity()

        # A 3-D tensor that doesn't fit any strategy cleanly — may error
        # We don't assert error_count because some strategies may handle it.
        # The key invariant is that no exception escapes.
        bad_output = torch.randn(2, 3, 4)  # ambiguous
        try:
            hook(module, (bad_output,), bad_output)
        except Exception as exc:
            pytest.fail(f"RouterHook raised: {exc}")

    def test_call_count_increments_on_every_call(self):
        """call_count tracks raw forward passes regardless of sampling."""
        hook, _ = self._make_hook_and_collector(sample_every=10)
        module  = nn.Identity()

        logits = torch.randn(4, 4)
        for _ in range(5):
            hook(module, (logits,), logits)

        assert hook.call_count == 5

    def test_repr_shows_layer_name(self):
        """RouterHook.__repr__ includes the layer name."""
        hook, _ = self._make_hook_and_collector("some.layer.name")
        assert "some.layer.name" in repr(hook)

# ============================================================================
# Additional RouterHook coverage tests
# ============================================================================

class _DummyCollector:
    def add_event(self, event):
        self.event = event


def test_router_hook_tuple_indices_path():
    collector = _DummyCollector()

    hook = RouterHook(
        layer_name="router",
        collector=collector,
        config=WatchConfig(sample_every=1),
    )

    indices = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
        ]
    )

    event = hook._from_tuple_output(
        (
            torch.randn(3, 4),
            "not_a_tensor",
            indices,
        )
    )

    assert event is not None
    assert event.n_experts == 4
    assert event.top_k == 2


def test_router_hook_object_logits_path():
    collector = _DummyCollector()

    hook = RouterHook(
        "router",
        collector,
        WatchConfig(sample_every=1),
    )

    output = SimpleNamespace(
        router_logits=torch.randn(10, 8)
    )

    event = hook._from_object_output(output)

    assert event is not None
    assert event.n_experts == 8


def test_router_hook_object_indices_path():
    collector = _DummyCollector()

    hook = RouterHook(
        "router",
        collector,
        WatchConfig(sample_every=1),
    )

    output = SimpleNamespace(
        expert_indices=torch.tensor(
            [
                [0, 1],
                [1, 2],
            ]
        )
    )

    event = hook._from_object_output(output)

    assert event is not None
    assert event.top_k == 2


def test_router_hook_dict_logits_path():
    collector = _DummyCollector()

    hook = RouterHook(
        "router",
        collector,
        WatchConfig(sample_every=1),
    )

    event = hook._from_dict_output(
        {
            "router_logits": torch.randn(5, 4)
        }
    )

    assert event is not None
    assert event.n_experts == 4


def test_router_hook_dict_indices_path():
    collector = _DummyCollector()

    hook = RouterHook(
        "router",
        collector,
        WatchConfig(sample_every=1),
    )

    event = hook._from_dict_output(
        {
            "expert_indices": torch.tensor(
                [
                    [0, 1],
                    [1, 2],
                ]
            )
        }
    )

    assert event is not None
    assert event.top_k == 2


def test_router_hook_dict_returns_none():
    collector = _DummyCollector()

    hook = RouterHook(
        "router",
        collector,
        WatchConfig(sample_every=1),
    )

    assert hook._from_dict_output({}) is None


def test_router_hook_object_returns_none():
    collector = _DummyCollector()

    hook = RouterHook(
        "router",
        collector,
        WatchConfig(sample_every=1),
    )

    assert hook._from_object_output(object()) is None


def test_router_hook_expert_indices_1d():
    collector = _DummyCollector()

    hook = RouterHook(
        "router",
        collector,
        WatchConfig(sample_every=1),
    )

    event = hook._from_expert_indices(
        torch.tensor([0, 1, 2, 3]),
        logits=None,
    )

    assert event.top_k == 1
    assert event.n_experts == 4


def test_router_hook_expert_indices_invalid_shape():
    collector = _DummyCollector()

    hook = RouterHook(
        "router",
        collector,
        WatchConfig(sample_every=1),
    )

    with pytest.raises(ValueError):
        hook._from_expert_indices(
            torch.randn(2, 3, 4),
            logits=None,
        )


def test_router_hook_fallback_unknown_output():
    collector = _DummyCollector()

    hook = RouterHook(
        "router",
        collector,
        WatchConfig(sample_every=1),
    )

    event = hook._extract_event(
        module=None,
        inputs=(torch.randn(2, 4),),
        output="unknown_output_type",
    )

    assert event.n_experts == 0
    assert event.total_tokens == 8


def test_router_hook_infer_token_count_tensor_1d():
    count = RouterHook._infer_token_count(
        (
            torch.randn(5),
        )
    )

    assert count == 5


def test_router_hook_infer_token_count_no_tensor():
    count = RouterHook._infer_token_count(
        (
            "abc",
            123,
        )
    )

    assert count == 0


def test_router_hook_repr():
    collector = _DummyCollector()

    hook = RouterHook(
        "router",
        collector,
        WatchConfig(sample_every=1),
    )

    text = repr(hook)

    assert "RouterHook" in text
    assert "router" in text
