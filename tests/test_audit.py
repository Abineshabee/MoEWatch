# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  tests/test_audit.py — Integration tests for moewatch._audit.audit()
#
#  COVERAGE TARGETS
#  ----------------
#    audit() — happy-path
#      ├── Returns an AuditReport when given a model + dataloader
#      ├── Returns an AuditReport when given a model only (synthetic mode)
#      ├── completed_steps == min(steps, len(dataloader))
#      ├── elapsed_seconds > 0
#      ├── router_module_names match the manually-specified list
#      ├── AuditReport.n_layers == len(router_modules)
#      ├── report.dead_experts() returns a list (empty or populated)
#      ├── report.routing_entropy() returns a dict keyed by layer name
#      ├── report.utilization() returns a dict keyed by layer name
#      ├── report.to_json() returns a valid JSON string
#      ├── report.summary() returns a non-empty string
#      └── report.recommendations() returns a list of strings
#
#    audit() — collapsed model detection
#      ├── Collapsed model → overall_health == CRITICAL
#      ├── dead_experts() non-empty for collapsed model
#      └── recommendations() non-empty for collapsed model
#
#    audit() — input validation
#      ├── TypeError for non-Module model
#      ├── ValueError for steps < 1
#      ├── ValueError for empty dataloader
#      ├── RuntimeError when no router modules detected and none specified
#      └── UserWarning when no dataloader is provided (synthetic mode)
#
#    audit() — configuration
#      ├── WatchConfig(output=SILENT) suppresses all stdout output
#      ├── Manual router_modules override bypasses auto-detection
#      ├── use_no_grad=False does not raise
#      └── Hooks are always detached after audit() (no hook leaks)
#
#    AuditReport query methods
#      ├── overall_health ∈ {HEALTHY, DEGRADING, CRITICAL, UNKNOWN}
#      ├── has_collapse True for collapsed model
#      ├── has_warnings is bool
#      ├── layer_names() returns router module names
#      ├── layer_stats(name) returns LayerStats or None
#      ├── entropy_for(name) returns EntropyResult or None
#      ├── collapse_for(name) returns LayerCollapseReport or None
#      ├── to_json_file() writes a valid file to disk
#      └── __bool__ is True when data was collected
#
#    Internal Helper Functions
#      ├── _resolve_device() CPU fallback for parameterless models
#      ├── _is_dataloader_empty() IterableDataset / TypeError branch
#      ├── _make_synthetic_batch() default synthetic batch generation
#      ├── _no_grad_ctx(False) execution path
#      ├── _infinite_iter() infinite iterator behavior
#      └── _print_audit_header() diagnostic banner rendering
#
#    _forward_pass() Branch Coverage
#      ├── Tensor batch input path
#      ├── Tuple/List batch input path
#      └── Unknown batch type warning path
#
#    audit() — Edge Cases
#      ├── Dataloader exhaustion warning
#      └── Verbose output/header rendering
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#
# =============================================================================

from __future__ import annotations

import json
import os
import tempfile
import warnings

import pytest
import torch

from conftest import SyntheticMoE, run_forward_passes

from moewatch import audit, WatchConfig
from moewatch.config import OutputMode
from moewatch.report.audit_report import AuditReport, OverallHealth


# =============================================================================
# §1  Happy-Path: audit() returns a valid AuditReport
# =============================================================================

class TestAuditHappyPath:

    def test_returns_audit_report_with_dataloader(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        """audit() with a real dataloader returns an AuditReport."""
        report = audit(
            synthetic_model,
            tiny_dataloader,
            steps=10,
            config=silent_config,
            verbose=False,
        )
        assert isinstance(report, AuditReport)

    def test_returns_audit_report_without_dataloader(
        self, synthetic_model, silent_config
    ):
        """audit() without a dataloader uses synthetic data — still returns AuditReport."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = audit(
                synthetic_model,
                steps=5,
                config=silent_config,
                verbose=False,
            )
        assert isinstance(report, AuditReport)

    def test_completed_steps_matches_requested(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        """completed_steps equals the requested steps when dataloader is large enough."""
        steps  = 10
        report = audit(
            synthetic_model,
            tiny_dataloader,
            steps=steps,
            config=silent_config,
            verbose=False,
        )
        assert report.completed_steps == steps

    def test_elapsed_seconds_positive(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=silent_config, verbose=False,
        )
        assert report.elapsed_seconds > 0.0

    def test_n_layers_matches_router_modules(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        """n_layers equals the number of manually-specified router modules."""
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=config, verbose=False,
        )
        assert report.n_layers == len(synthetic_model.router_module_names)

    def test_router_module_names_match_config(
        self, synthetic_model, tiny_dataloader
    ):
        """report.router_module_names matches the names in WatchConfig."""
        names  = synthetic_model.router_module_names
        config = WatchConfig(
            router_modules=names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=config, verbose=False,
        )
        assert report.router_module_names == names

    def test_device_attribute_is_string(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=silent_config, verbose=False,
        )
        assert isinstance(report.device, str)

    def test_dead_experts_returns_list(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=silent_config, verbose=False,
        )
        assert isinstance(report.dead_experts(), list)

    def test_routing_entropy_returns_dict(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=config, verbose=False,
        )
        entropy = report.routing_entropy()
        assert isinstance(entropy, dict)
        for name in synthetic_model.router_module_names:
            assert name in entropy

    def test_utilization_returns_dict(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=config, verbose=False,
        )
        util = report.utilization()
        assert isinstance(util, dict)

    def test_to_json_returns_valid_json(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=config, verbose=False,
        )
        json_str = report.to_json()
        parsed   = json.loads(json_str)
        assert "metadata" in parsed
        assert "entropy" in parsed
        assert "collapse" in parsed

    def test_summary_returns_non_empty_string(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=config, verbose=False,
        )
        s = report.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_recommendations_returns_list_of_strings(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=config, verbose=False,
        )
        recs = report.recommendations()
        assert isinstance(recs, list)
        assert all(isinstance(r, str) for r in recs)

    def test_bool_is_true_when_data_collected(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=config, verbose=False,
        )
        assert bool(report) is True


# =============================================================================
# §2  Collapsed Model — Detection End-to-End
# =============================================================================

class TestAuditCollapseDetection:

    def _audit_collapsed(self, tiny_dataloader):
        """Run audit on a model that forces collapse on layer 0."""
        model  = SyntheticMoE(n_layers=2, n_experts=8, hidden_dim=64, collapse_layer=0)
        config = WatchConfig(
            router_modules=model.router_module_names,
            dead_threshold=0.001,
            cold_threshold=0.005,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        return audit(model, tiny_dataloader, steps=20, config=config, verbose=False)

    def test_collapsed_model_overall_health_critical(self, tiny_dataloader):
        """A collapsed model should have CRITICAL overall health."""
        report = self._audit_collapsed(tiny_dataloader)
        assert report.overall_health == OverallHealth.CRITICAL

    def test_collapsed_model_has_collapse_true(self, tiny_dataloader):
        report = self._audit_collapsed(tiny_dataloader)
        assert report.has_collapse is True

    def test_collapsed_model_dead_experts_non_empty(self, tiny_dataloader):
        """dead_experts() is non-empty for a collapsed model."""
        report = self._audit_collapsed(tiny_dataloader)
        dead   = report.dead_experts(include_cold=False)
        assert len(dead) >= 1

    def test_collapsed_model_recommendations_non_empty(self, tiny_dataloader):
        """Collapsed model produces at least one recommendation."""
        report = self._audit_collapsed(tiny_dataloader)
        recs   = report.recommendations()
        # At minimum, a [CRITICAL] dead expert recommendation should appear
        assert any("[CRITICAL]" in r for r in recs)

    def test_healthy_model_overall_health_healthy(
        self, synthetic_model, tiny_dataloader
    ):
        """A uniform-routing model should not be CRITICAL."""
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=20, config=config, verbose=False,
        )
        # Healthy model should be HEALTHY or at worst DEGRADING (never CRITICAL)
        assert report.overall_health in (OverallHealth.HEALTHY, OverallHealth.DEGRADING)

    def test_n_dead_equals_dead_experts_list_length(self, tiny_dataloader):
        report = self._audit_collapsed(tiny_dataloader)
        assert report.n_dead == len(report.dead_experts(include_cold=False))


# =============================================================================
# §3  Input Validation
# =============================================================================

class TestAuditInputValidation:

    def test_non_module_raises_type_error(self, tiny_dataloader, silent_config):
        with pytest.raises(TypeError, match="torch.nn.Module"):
            audit("not a model", tiny_dataloader, steps=5, config=silent_config)

    def test_steps_less_than_one_raises_value_error(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        with pytest.raises(ValueError, match="steps"):
            audit(synthetic_model, tiny_dataloader, steps=0, config=silent_config)

    def test_steps_negative_raises_value_error(
        self, synthetic_model, tiny_dataloader, silent_config
    ):
        with pytest.raises(ValueError, match="steps"):
            audit(synthetic_model, tiny_dataloader, steps=-1, config=silent_config)

    def test_empty_dataloader_raises_value_error(
        self, synthetic_model, silent_config
    ):
        with pytest.raises(ValueError, match="empty"):
            audit(synthetic_model, [], steps=5, config=silent_config)

    def test_no_router_modules_detected_raises_runtime_error(self):
        """A dense model with no routers and no manual override → RuntimeError."""
        import torch.nn as nn
        from moewatch.config import WatchConfig, OutputMode

        class _DenseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
            def forward(self, x):
                return self.linear(x)

        dense         = _DenseModel()
        batches       = [{"input_ids": torch.randint(0, 10, (1, 4))} for _ in range(5)]
        clean_config  = WatchConfig(output=OutputMode.SILENT)   # no router_modules set

        with pytest.raises(RuntimeError, match="router"):
            audit(dense, batches, steps=3, config=clean_config)

    def test_no_dataloader_emits_user_warning(
        self, synthetic_model, silent_config
    ):
        """Omitting the dataloader triggers a UserWarning about synthetic data."""
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            audit(synthetic_model, steps=3, config=config, verbose=False)

        assert any(issubclass(warning.category, UserWarning) for warning in w)


# =============================================================================
# §4  Configuration and Lifecycle
# =============================================================================

class TestAuditConfiguration:

    def test_manual_router_modules_bypasses_detection(
        self, synthetic_model, tiny_dataloader
    ):
        """Manually-specified router_modules are used as-is."""
        names  = synthetic_model.router_module_names
        config = WatchConfig(
            router_modules=names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=5, config=config, verbose=False,
        )
        assert set(report.router_module_names) == set(names)

    def test_hooks_detached_after_audit(
        self, synthetic_model, tiny_dataloader
    ):
        """No forward hooks remain on the model after audit() completes."""
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        audit(synthetic_model, tiny_dataloader, steps=5, config=config, verbose=False)

        # Check that no _forward_hooks remain on any submodule
        for module in synthetic_model.modules():
            assert len(module._forward_hooks) == 0, (
                f"Leaked forward hook on {type(module).__name__}"
            )

    def test_hooks_detached_even_after_error(
        self, synthetic_model
    ):
        """Hooks are cleaned up even when the forward pass raises an exception."""
        config  = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )

        class _BadBatch:
            """Iterator that raises after the first batch."""
            def __init__(self):
                self._count = 0
            def __iter__(self):
                return self
            def __next__(self):
                if self._count == 0:
                    self._count += 1
                    return {"input_ids": torch.randint(0, 256, (1, 8))}
                raise RuntimeError("intentional error in dataloader")
            def __len__(self):
                return 10

        try:
            audit(synthetic_model, _BadBatch(), steps=5, config=config, verbose=False)
        except Exception:
            pass  # exception is expected — we only care about hook cleanup

        for module in synthetic_model.modules():
            assert len(module._forward_hooks) == 0

    def test_use_no_grad_false_does_not_raise(
        self, synthetic_model, tiny_dataloader
    ):
        """use_no_grad=False allows gradient flow — should not raise."""
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        # Will only work if no in-place operations conflict with autograd
        report = audit(
            synthetic_model, tiny_dataloader,
            steps=3, config=config,
            use_no_grad=False, verbose=False,
        )
        assert isinstance(report, AuditReport)

    def test_verbose_false_produces_no_stdout(
        self, synthetic_model, tiny_dataloader, capsys
    ):
        """verbose=False suppresses all stdout output."""
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        audit(synthetic_model, tiny_dataloader, steps=5, config=config, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""


# =============================================================================
# §5  AuditReport Per-Layer Accessors
# =============================================================================

class TestAuditReportAccessors:

    @pytest.fixture()
    def base_report(self, synthetic_model, tiny_dataloader):
        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )
        return audit(
            synthetic_model, tiny_dataloader,
            steps=10, config=config, verbose=False,
        )

    def test_overall_health_is_valid_enum(self, base_report):
        assert base_report.overall_health in list(OverallHealth)

    def test_has_warnings_is_bool(self, base_report):
        assert isinstance(base_report.has_warnings, bool)

    def test_n_experts_total_positive(self, base_report):
        assert base_report.n_experts_total > 0

    def test_layer_names_matches_router_module_names(
        self, synthetic_model, base_report
    ):
        assert base_report.layer_names() == synthetic_model.router_module_names

    def test_layer_stats_valid_for_known_name(
        self, synthetic_model, base_report
    ):
        name  = synthetic_model.router_module_names[0]
        stats = base_report.layer_stats(name)
        assert stats is not None

    def test_layer_stats_none_for_unknown_name(self, base_report):
        assert base_report.layer_stats("nonexistent.layer") is None

    def test_entropy_for_valid_layer(self, synthetic_model, base_report):
        name   = synthetic_model.router_module_names[0]
        result = base_report.entropy_for(name)
        assert result is not None

    def test_entropy_for_unknown_layer_returns_none(self, base_report):
        assert base_report.entropy_for("nonexistent") is None

    def test_collapse_for_valid_layer(self, synthetic_model, base_report):
        name   = synthetic_model.router_module_names[0]
        report = base_report.collapse_for(name)
        assert report is not None

    def test_collapse_for_unknown_layer_returns_none(self, base_report):
        assert base_report.collapse_for("nonexistent") is None

    def test_worst_entropy_layer_is_none_or_entropy_result(self, base_report):
        worst = base_report.worst_entropy_layer
        # Can be None (if no data) or an EntropyResult
        if worst is not None:
            from moewatch.analyzer.entropy import EntropyResult
            assert isinstance(worst, EntropyResult)

    def test_to_json_file_writes_readable_json(self, base_report, tmp_path):
        path = str(tmp_path / "test_report.json")
        base_report.to_json_file(path)

        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "metadata" in data

    def test_repr_contains_health_and_layers(self, base_report):
        r = repr(base_report)
        assert "health=" in r
        assert "layers=" in r

    def test_dead_experts_include_cold_false_subset_of_all(self, base_report):
        """dead_experts(include_cold=False) is a subset of dead_experts(include_cold=True)."""
        all_dead  = base_report.dead_experts(include_cold=True)
        only_dead = base_report.dead_experts(include_cold=False)
        assert len(only_dead) <= len(all_dead)
        all_dead_idxs  = {(e.layer_name, e.expert_idx) for e in all_dead}
        only_dead_idxs = {(e.layer_name, e.expert_idx) for e in only_dead}
        assert only_dead_idxs.issubset(all_dead_idxs)


# =============================================================================
# Extra Coverage Tests for moewatch._audit
# =============================================================================

import warnings
import torch
import torch.nn as nn

from moewatch._audit import (
    _resolve_device,
    _is_dataloader_empty,
    _make_synthetic_batch,
    _forward_pass,
    _no_grad_ctx,
    _infinite_iter,
    _print_audit_header,
)


class TestAuditInternalHelpers:

    def test_resolve_device_no_parameters_returns_cpu(self):
        class EmptyModel(nn.Module):
            def forward(self, x):
                return x

        model = EmptyModel()
        device = _resolve_device(model)

        assert str(device) == "cpu"

    def test_is_dataloader_empty_typeerror_branch(self):
        class IterableOnly:
            def __iter__(self):
                yield 1

        assert _is_dataloader_empty(IterableOnly()) is False

    def test_make_synthetic_batch_default_vocab(self):
        class NoEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        batch = _make_synthetic_batch(
            NoEmbedding(),
            torch.device("cpu"),
            seq_len=8,
            batch_size=2,
        )

        assert batch["input_ids"].shape == (2, 8)
        assert batch["attention_mask"].shape == (2, 8)

    def test_no_grad_ctx_false_branch(self):
        with _no_grad_ctx(False):
            x = torch.tensor([1.0], requires_grad=True)

        assert x.requires_grad

    def test_infinite_iter(self):
        gen = _infinite_iter("hello")

        assert next(gen) == "hello"
        assert next(gen) == "hello"
        assert next(gen) == "hello"

    def test_print_audit_header(self, capsys):
        from moewatch import WatchConfig

        _print_audit_header(
            n_routers=2,
            steps=10,
            device=torch.device("cpu"),
            config=WatchConfig(),
        )

        out = capsys.readouterr().out

        assert "Offline Diagnostic Audit" in out
        assert "Router modules detected" in out


class TestForwardPassBranches:

    def test_forward_pass_tensor(self):
        class TensorModel(nn.Module):
            def forward(self, x):
                return x

        model = TensorModel()

        batch = torch.randn(2, 3)

        _forward_pass(
            model,
            batch,
            torch.device("cpu"),
        )

    def test_forward_pass_tuple(self):
        class TupleModel(nn.Module):
            def forward(self, x, y):
                return x + y

        model = TupleModel()

        batch = (
            torch.ones(2, 2),
            torch.ones(2, 2),
        )

        _forward_pass(
            model,
            batch,
            torch.device("cpu"),
        )

    def test_forward_pass_unknown_batch_warns(self):
        class EchoModel(nn.Module):
            def forward(self, x):
                return x

        model = EchoModel()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _forward_pass(
                model,
                "unknown-batch",
                torch.device("cpu"),
            )

        assert any(
            "Unrecognised batch type" in str(x.message)
            for x in w
        )


class TestAuditAdditionalBranches:

    def test_dataloader_exhaustion_warning(
        self,
        synthetic_model,
    ):
        from moewatch import WatchConfig
        from moewatch.config import OutputMode
        from moewatch import audit

        batches = [
            {"input_ids": torch.randint(0, 100, (1, 8))}
        ]

        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            output=OutputMode.SILENT,
            sample_every=1,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            audit(
                synthetic_model,
                batches,
                steps=20,
                config=config,
                verbose=False,
            )

        assert any(
            "Dataloader exhausted" in str(x.message)
            for x in w
        )

    def test_verbose_output_branch(
        self,
        synthetic_model,
        tiny_dataloader,
        capsys,
    ):
        from moewatch import WatchConfig
        from moewatch import audit

        config = WatchConfig(
            router_modules=synthetic_model.router_module_names,
            sample_every=1,
        )

        audit(
            synthetic_model,
            tiny_dataloader,
            steps=2,
            config=config,
            verbose=True,
        )

        out = capsys.readouterr().out

        assert "Offline Diagnostic Audit" in out
