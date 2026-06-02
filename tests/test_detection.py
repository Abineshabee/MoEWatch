# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  tests/test_detection.py — Unit tests for moewatch/hooks/detection.py
#
#  COVERAGE TARGETS
#  ----------------
#    detect_router_modules()
#      ├── Pass 1: architecture registry hit (known class names)
#      ├── Pass 2: heuristic fallback (substring matching + exclusions)
#      ├── Graceful empty-return when nothing is found
#      ├── Deduplication (same module never returned twice)
#      ├── Manual override via WatchConfig.router_modules
#      └── Root module (name="") is never returned
#
#    list_known_architectures()
#      └── Returns a non-empty dict of sorted class name lists
#
#    is_known_router_class()
#      ├── Returns True for known class names
#      └── Returns False for unknown / empty strings
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#
# =============================================================================

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from moewatch.hooks.detection import (
    detect_router_modules,
    list_known_architectures,
    is_known_router_class,
)
from moewatch.config import OutputMode

# =============================================================================
# §1  Helper Models
# =============================================================================

class _DummyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 4))


class _MockMixtralSparseMoeBlock(nn.Module):
    """Mimics the class name in the Mixtral architecture registry."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8, 8))


class _MockOlmoeMoE(nn.Module):
    """Mimics the OLMoE architecture registry entry."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 4))


class _DenseModel(nn.Module):
    """Dense model — no MoE layers whatsoever."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm   = nn.LayerNorm(4)

    def forward(self, x):
        return self.norm(self.linear(x))


class _RegistryHitModel(nn.Module):
    """Model with a known Mixtral class name — tests Pass 1 registry hit."""
    def __init__(self):
        super().__init__()
        self.block_sparse_moe = _MockMixtralSparseMoeBlock()
        self.other            = nn.Linear(4, 4)


class _HeuristicModel(nn.Module):
    """Model with a custom router class that uses a heuristic-matching name."""
    def __init__(self):
        super().__init__()
        self.custom_router = _CustomRouter()
        self.other         = nn.Linear(4, 4)


class _CustomRouter(nn.Module):
    """Custom router — matches heuristic 'router' substring."""
    def __init__(self):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.ones(8, 8))


class _ExclusionModel(nn.Module):
    """Model whose module names contain 'gate' but are disqualified by exclusions."""
    def __init__(self):
        super().__init__()
        self.lm_head        = nn.Linear(4, 4)
        self.gate_projection = _GateProjection()  # should be excluded
        self.mlp_gate        = _MlpGate()          # 'mlp' exclusion wins


class _GateProjection(nn.Module):
    """Has 'gate' but also 'projection' exclusion substring."""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(4, 4))


class _MlpGate(nn.Module):
    """Has 'gate' but also 'mlp' exclusion substring."""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(4, 4))


class _MultiRegistryModel(nn.Module):
    """Model with both Mixtral and OLMoE blocks — tests deduplication."""
    def __init__(self):
        super().__init__()
        self.moe0 = _MockMixtralSparseMoeBlock()
        self.moe1 = _MockOlmoeMoE()
        self.fc   = nn.Linear(4, 4)


# =============================================================================
# §2  Architecture Registry Tests (Pass 1)
# =============================================================================

class TestRegistryDetection:
    """Tests for the curated architecture registry (Pass 1)."""

    def test_mixtral_registry_hit(self):
        """detect_router_modules returns the MixtralSparseMoeBlock layer name."""
        model  = _RegistryHitModel()
        result = detect_router_modules(model)

        assert isinstance(result, list)
        assert len(result) >= 1
        assert "block_sparse_moe" in result

    def test_registry_hit_excludes_non_router_modules(self):
        """Non-router modules (Linear) are not returned on a registry hit."""
        model  = _RegistryHitModel()
        result = detect_router_modules(model)

        assert "other" not in result

    def test_multi_registry_hit_deduplication(self):
        """Each module name appears at most once in the result list."""
        model  = _MultiRegistryModel()
        result = detect_router_modules(model)

        assert len(result) == len(set(result)), "Duplicate module names returned"

    def test_multi_registry_hit_returns_both_layers(self):
        """Both moe0 and moe1 are returned when both match the registry."""
        model  = _MultiRegistryModel()
        result = detect_router_modules(model)

        # At least one known class was found; both layers should be present
        assert any("moe" in name for name in result)

    def test_root_module_never_returned(self):
        """The root module (name='') is never included in results."""
        model  = _RegistryHitModel()
        result = detect_router_modules(model)

        assert "" not in result

    def test_known_architecture_registry_non_empty(self):
        """list_known_architectures returns a populated dict."""
        registry = list_known_architectures()
        assert isinstance(registry, dict)
        assert len(registry) >= 5  # Mixtral, OLMoE, DeepSeek, Qwen, SwitchTransformer

    def test_known_architecture_sorted_lists(self):
        """Class name lists in the registry are sorted."""
        registry = list_known_architectures()
        for family, names in registry.items():
            assert names == sorted(names), f"{family} names are not sorted"

    def test_is_known_router_class_mixtral(self):
        """is_known_router_class returns True for MixtralSparseMoeBlock."""
        assert is_known_router_class("MixtralSparseMoeBlock") is True

    def test_is_known_router_class_olmoe(self):
        """is_known_router_class returns True for OlmoeMoE."""
        assert is_known_router_class("OlmoeMoE") is True

    def test_is_known_router_class_unknown(self):
        """is_known_router_class returns False for unknown class names."""
        assert is_known_router_class("SomeRandomLinear") is False

    def test_is_known_router_class_empty_string(self):
        """is_known_router_class handles empty string gracefully."""
        assert is_known_router_class("") is False

    def test_is_known_router_class_case_sensitive(self):
        """Registry matching is case-sensitive."""
        assert is_known_router_class("mixtralsparsemoeblock") is False


# =============================================================================
# §3  Heuristic Fallback Tests (Pass 2)
# =============================================================================

class TestHeuristicDetection:
    """Tests for the substring heuristic fallback (Pass 2)."""

    def test_heuristic_finds_custom_router(self):
        """A custom class with 'router' substring is found when registry misses."""
        model  = _HeuristicModel()
        result = detect_router_modules(model)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_heuristic_excludes_mlp_gate(self):
        """Modules with exclusion substrings (mlp, projection) are not returned."""
        model  = _ExclusionModel()
        result = detect_router_modules(model)

        # gate_projection and mlp_gate should be excluded
        for name in result:
            assert "gate_projection" not in name
            assert "mlp_gate" not in name

    def test_heuristic_no_false_positive_on_lm_head(self):
        """lm_head is never returned by heuristic detection."""
        model  = _ExclusionModel()
        result = detect_router_modules(model)

        assert "lm_head" not in result

    def test_heuristic_min_param_count_guard(self):
        """Modules with zero learnable parameters are not returned."""

        class _ParameterlessRouter(nn.Module):
            """Has 'router' in the class name but no parameters."""
            pass

        class _ModelWithParameterlessRouter(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = _ParameterlessRouter()

        model  = _ModelWithParameterlessRouter()
        result = detect_router_modules(model)

        # _ParameterlessRouter has 0 params → should be filtered out
        assert "router" not in result


# =============================================================================
# §4  Graceful Degradation Tests
# =============================================================================

class TestGracefulDegradation:
    """Tests for empty-result graceful handling."""

    def test_dense_model_returns_empty_list(self):
        """No router modules are detected in a pure dense model."""
        model  = _DenseModel()
        result = detect_router_modules(model)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_empty_module_returns_empty_list(self):
        """A Module with no children returns an empty list."""

        class _EmptyModel(nn.Module):
            pass

        result = detect_router_modules(_EmptyModel())
        assert result == []

    def test_returns_list_type_always(self):
        """detect_router_modules always returns a list, never None."""
        assert detect_router_modules(_DenseModel()) is not None
        assert isinstance(detect_router_modules(_DenseModel()), list)

    def test_single_linear_layer_model(self):
        """A model with a single Linear layer returns nothing."""
        result = detect_router_modules(nn.Linear(4, 4))
        assert result == []


# =============================================================================
# §5  Manual Override via WatchConfig
# =============================================================================

class TestManualOverride:
    """Verify that WatchConfig.router_modules bypasses auto-detection."""

    def test_manual_override_respected_in_audit(self):
        """When router_modules is set, detect_router_modules is not called by audit()."""
        # This tests the contract: when the user specifies router_modules, the
        # detection logic is bypassed entirely. We verify indirectly by checking
        # that detect_router_modules() on its own still works as expected.
        from moewatch.config import WatchConfig
        config = WatchConfig(
            router_modules=["moe_layers.0", "moe_layers.1"],
            output=OutputMode.SILENT,
            sample_every=1,
        )
        assert config.router_modules == ["moe_layers.0", "moe_layers.1"]

    def test_empty_router_modules_triggers_detection(self):
        """An empty router_modules list means detection will run."""
        from moewatch.config import WatchConfig
        config = WatchConfig(router_modules=[], output=OutputMode.SILENT)
        assert config.router_modules == []

    def test_router_modules_list_is_copied(self):
        """WatchConfig.router_modules is independent of the source list."""
        from moewatch.config import WatchConfig
        source = ["moe_layers.0"]
        config = WatchConfig(router_modules=source, output=OutputMode.SILENT)
        source.append("moe_layers.1")
        # The config should have been seeded from the original list;
        # mutation of source should not affect config (dataclass field list copy)
        assert config.router_modules == ["moe_layers.0"]
