# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  tests/conftest.py — Shared pytest fixtures for the moewatch test suite
#
#  SCOPE
#  -----
#  All fixtures defined here are automatically discovered by pytest in every
#  test module under tests/. Nothing in this file prints to stdout; all
#  output is captured by pytest's default capture mechanism.
#
#  DESIGN PHILOSOPHY
#  -----------------
#  Every fixture is GPU-free and torch-only. No HuggingFace weights are
#  downloaded. The synthetic model is intentionally tiny so tests complete
#  in milliseconds on CI. All fixtures follow the principle:
#    "if it can be built in under 50 ms, build it fresh each test".
#  Expensive objects (e.g. multi-layer model) use function scope so isolation
#  is guaranteed.
#
#  FIXTURE INVENTORY
#  -----------------
#    default_config        — WatchConfig with safe test-friendly defaults.
#    aggressive_config     — WatchConfig with tight thresholds for collapse tests.
#    silent_config         — WatchConfig(output=SILENT), suppresses all output.
#    synthetic_router      — Minimal single-layer SyntheticMoE model.
#    synthetic_model       — Multi-layer SyntheticMoE (8 layers, 8 experts each).
#    multilayer_model      — Alias for synthetic_model with explicit name.
#    tiny_dataloader       — Iterable of 20 random dict batches.
#    mock_routing_event    — Factory for RoutingEvent with configurable counts.
#    collapsed_routing_event — RoutingEvent with one dead expert.
#    uniform_routing_event — RoutingEvent with perfectly uniform distribution.
#    stat_collector_factory  — Factory: StatCollector pre-seeded with events.
#    layer_stats_healthy   — LayerStats representing a healthy layer.
#    layer_stats_collapsed — LayerStats with an expert at 0 % utilisation.
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#  Version: 0.1.0
#
# =============================================================================

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Internal moewatch imports — resolved against the package on sys.path.
# CI installs moewatch via ``pip install -e .`` before running pytest.
# ---------------------------------------------------------------------------

from moewatch.config import WatchConfig, OutputMode, AlertLevel
from moewatch.hooks.router_hook import RoutingEvent
from moewatch.collector.stat_collector import StatCollector, LayerStats


# =============================================================================
# §1  SYNTHETIC MOE ARCHITECTURE
#
# SyntheticMoERouter — a minimal MoE router that mimics Mixtral's structure:
#   inputs  → gate_proj → softmax → select top-k → route to n expert MLPs
#   output  → (hidden_states, router_logits)  [tuple, like Mixtral]
#
# The forward hook in moewatch's RouterHook._from_tuple_output handles this
# because output[1] is a 2-D logit tensor of shape (batch*seq_len, n_experts).
# =============================================================================

class SyntheticExpert(nn.Module):
    """Minimal expert MLP. Hidden-dim → hidden-dim identity transform."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SyntheticMoERouter(nn.Module):
    """Single MoE router layer mimicking Mixtral's MixtralSparseMoeBlock.

    Returns (hidden_states, router_logits) — a 2-tuple where index [1] is
    shape (batch * seq_len, n_experts). RouterHook._from_tuple_output handles
    this pattern (Strategy 1 in router_hook.py).

    Parameters
    ----------
    hidden_dim : int
    n_experts  : int
    top_k      : int
    collapse_expert : int | None
        If set, forces this expert index to always receive 100 % of traffic,
        inducing a synthetic collapse for testing CollapseDetector.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_experts: int = 8,
        top_k: int = 1,
        collapse_expert: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_experts       = n_experts
        self.top_k           = top_k
        self.collapse_expert = collapse_expert

        # Gate produces router logits: (batch*seq, n_experts)
        self.gate    = nn.Linear(hidden_dim, n_experts, bias=False)
        self.experts = nn.ModuleList([SyntheticExpert(hidden_dim) for _ in range(n_experts)])

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        hidden_states : (batch, seq_len, hidden_dim)

        Returns
        -------
        (hidden_states, router_logits)
            router_logits shape: (batch * seq_len, n_experts)
        """
        batch, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(batch * seq_len, hidden_dim)

        router_logits = self.gate(flat)  # (B*S, n_experts)

        if self.collapse_expert is not None:
            # Force one expert to dominate — simulate collapse
            with torch.no_grad():
                router_logits.fill_(-1e9)
                router_logits[:, self.collapse_expert] = 1e9

        probs   = torch.softmax(router_logits, dim=-1)
        _, indices = torch.topk(probs, self.top_k, dim=-1)

        # Route through selected experts (simple sum for efficiency)
        output = torch.zeros_like(flat)
        for i in range(self.top_k):
            expert_idx = indices[:, i]
            for eidx in range(self.n_experts):
                mask = (expert_idx == eidx)
                if mask.any():
                    output[mask] += self.experts[eidx](flat[mask])

        out = output.reshape(batch, seq_len, hidden_dim)
        return out, router_logits  # tuple — RouterHook Strategy 1


class SyntheticMoE(nn.Module):
    """Multi-layer MoE model for end-to-end testing.

    Each layer is named ``moe_layer_{i}`` to avoid matching the architecture
    registry (which checks for Mixtral / OLMoE class names), keeping tests
    independent of the registry.

    Parameters
    ----------
    n_layers   : int   Number of MoE layers.
    n_experts  : int   Experts per layer.
    hidden_dim : int   Hidden dimension.
    vocab_size : int   Vocabulary for the embedding.
    collapse_layer : int | None
        If set, forces that layer's router into collapse mode.
    """

    def __init__(
        self,
        n_layers: int = 2,
        n_experts: int = 8,
        hidden_dim: int = 64,
        vocab_size: int = 256,
        collapse_layer: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding  = nn.Embedding(vocab_size, hidden_dim)

        self.moe_layers = nn.ModuleList([
            SyntheticMoERouter(
                hidden_dim=hidden_dim,
                n_experts=n_experts,
                collapse_expert=(0 if collapse_layer is not None and i == collapse_layer else None),
            )
            for i in range(n_layers)
        ])

        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass — returns logits of shape (batch, seq_len, vocab_size)."""
        x = self.embedding(input_ids)           # (batch, seq, hidden)

        for moe_layer in self.moe_layers:
            x, _router_logits = moe_layer(x)    # hooks fire here

        return self.lm_head(x)

    @property
    def router_module_names(self) -> List[str]:
        """Fully-qualified names of MoE router layers (mirrors WatchConfig override)."""
        return [f"moe_layers.{i}" for i in range(len(self.moe_layers))]


# =============================================================================
# §2  WATCH CONFIG FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def default_config() -> WatchConfig:
    """Default WatchConfig with test-safe ring buffer and sampling settings."""
    return WatchConfig(
        output=OutputMode.SILENT,
        sample_every=1,          # capture every forward pass in tests
        log_every=1,
        ring_buffer_capacity=1_000,
        window_steps=200,
    )


@pytest.fixture(scope="function")
def aggressive_config() -> WatchConfig:
    """WatchConfig with tight thresholds — triggers alerts faster in tests."""
    return WatchConfig(
        dead_threshold=0.05,      # 5% — easy to trigger in tests
        cold_threshold=0.10,      # 10%
        cold_steps_limit=2,       # promote to DEAD after 2 cold steps
        entropy_warn=0.80,
        entropy_critical=0.60,
        output=OutputMode.SILENT,
        sample_every=1,
        log_every=1,
        ring_buffer_capacity=500,
        window_steps=100,
    )


@pytest.fixture(scope="function")
def silent_config(synthetic_model) -> WatchConfig:
    """WatchConfig with SILENT output and pre-set router modules for the synthetic model."""
    return WatchConfig(
        output=OutputMode.SILENT,
        sample_every=1,
        log_every=1,
        router_modules=synthetic_model.router_module_names,
    )


# =============================================================================
# §3  MODEL FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def synthetic_router(default_config) -> SyntheticMoERouter:
    """Single SyntheticMoERouter layer — used for hook-level tests."""
    return SyntheticMoERouter(hidden_dim=64, n_experts=8)


@pytest.fixture(scope="function")
def synthetic_model() -> SyntheticMoE:
    """2-layer SyntheticMoE — the primary test model.

    Router module names are ``["moe_layers.0", "moe_layers.1"]``.
    """
    return SyntheticMoE(n_layers=2, n_experts=8, hidden_dim=64)


@pytest.fixture(scope="function")
def multilayer_model() -> SyntheticMoE:
    """4-layer model for tests that need more routing layers."""
    return SyntheticMoE(n_layers=4, n_experts=8, hidden_dim=64)


@pytest.fixture(scope="function")
def collapsed_model() -> SyntheticMoE:
    """2-layer model where layer 0 is always collapsed to expert 0."""
    return SyntheticMoE(n_layers=2, n_experts=8, hidden_dim=64, collapse_layer=0)


# =============================================================================
# §4  DATALOADER FIXTURES
# =============================================================================

def _make_batch(batch_size: int = 2, seq_len: int = 16, vocab_size: int = 256) -> Dict:
    """Build a single dict batch of random token ids."""
    input_ids      = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture(scope="function")
def tiny_dataloader() -> List[Dict]:
    """20 small dict batches — sufficient for end-to-end audit tests."""
    return [_make_batch() for _ in range(20)]


@pytest.fixture(scope="function")
def single_batch() -> Dict:
    """A single batch for minimal forward-pass tests."""
    return _make_batch(batch_size=1, seq_len=8)


# =============================================================================
# §5  ROUTING EVENT FACTORIES
# =============================================================================

def _make_routing_event(
    layer_name: str = "moe_layers.0",
    n_experts: int = 8,
    step: int = 1,
    expert_counts: Optional[List[int]] = None,
    total_tokens: Optional[int] = None,
    raw_logits: Optional[torch.Tensor] = None,
) -> RoutingEvent:
    """Build a RoutingEvent with configurable expert counts."""
    if expert_counts is None:
        # Uniform distribution
        per_expert = 100
        expert_counts = [per_expert] * n_experts
    counts_tensor = torch.tensor(expert_counts, dtype=torch.int64)
    if total_tokens is None:
        total_tokens = int(counts_tensor.sum().item())
    return RoutingEvent(
        layer_name=layer_name,
        step=step,
        timestamp=0.0,
        n_experts=n_experts,
        expert_counts=counts_tensor,
        total_tokens=total_tokens,
        raw_logits=raw_logits,
    )


@pytest.fixture(scope="function")
def mock_routing_event() -> RoutingEvent:
    """Uniform RoutingEvent — 100 tokens per expert across 8 experts."""
    return _make_routing_event()


@pytest.fixture(scope="function")
def collapsed_routing_event() -> RoutingEvent:
    """RoutingEvent where expert 0 gets 100% of tokens; all others get 0."""
    counts = [800] + [0] * 7  # expert 0 = 100%, rest = 0%
    return _make_routing_event(expert_counts=counts)


@pytest.fixture(scope="function")
def uniform_routing_event() -> RoutingEvent:
    """Alias for mock_routing_event (explicitly named for readability)."""
    return _make_routing_event()


@pytest.fixture(scope="function")
def cold_expert_routing_event() -> RoutingEvent:
    """RoutingEvent with expert 7 in COLD zone (0.3% — above dead, below cold)."""
    # cold_threshold default = 0.5%; we put expert 7 at 0.3%
    counts = [125, 125, 125, 125, 125, 124, 124, 3]  # expert 7 = 3/1000 = 0.3%
    return _make_routing_event(expert_counts=counts)


# =============================================================================
# §6  LAYER STATS FACTORIES
# =============================================================================

def _make_layer_stats(
    layer_name: str = "moe_layers.0",
    n_experts: int = 8,
    expert_counts: Optional[List[int]] = None,
    has_raw_logits: bool = False,
    raw_logits_window: Optional[list] = None,
    event_count: int = 10,
    step_range: tuple = (0, 9),
) -> LayerStats:
    """Build a LayerStats object with explicit expert counts."""
    if expert_counts is None:
        expert_counts = [100] * n_experts

    counts_tensor = torch.tensor(expert_counts, dtype=torch.int64)
    total_tokens  = int(counts_tensor.sum().item())

    if total_tokens > 0:
        utilization = counts_tensor.float() / total_tokens
        mean_util   = utilization.mean().item()
        max_util    = utilization.max().item()
        load_imbalance = max_util / mean_util if mean_util > 0 else float("nan")
    else:
        utilization    = torch.zeros(n_experts, dtype=torch.float32)
        load_imbalance = float("nan")

    return LayerStats(
        layer_name=layer_name,
        n_experts=n_experts,
        expert_counts=counts_tensor,
        total_tokens=total_tokens,
        utilization=utilization,
        load_imbalance_score=load_imbalance,
        top_k=1,
        event_count=event_count,
        step_range=step_range,
        has_raw_logits=has_raw_logits,
        raw_logits_window=raw_logits_window,
    )


@pytest.fixture(scope="function")
def layer_stats_healthy() -> LayerStats:
    """Perfectly balanced LayerStats — each of 8 experts gets 12.5% of tokens."""
    return _make_layer_stats(expert_counts=[100] * 8)


@pytest.fixture(scope="function")
def layer_stats_collapsed() -> LayerStats:
    """LayerStats where expert 7 is completely dead (0 tokens)."""
    counts = [150, 150, 150, 150, 150, 100, 100, 0]
    return _make_layer_stats(expert_counts=counts)


@pytest.fixture(scope="function")
def layer_stats_all_dead() -> LayerStats:
    """LayerStats where experts 4–7 are all at zero (dead)."""
    counts = [250, 250, 250, 250, 0, 0, 0, 0]
    return _make_layer_stats(expert_counts=counts)


@pytest.fixture(scope="function")
def layer_stats_one_hot() -> LayerStats:
    """LayerStats where expert 0 gets 100% of tokens — maximum collapse."""
    counts = [1000] + [0] * 7
    return _make_layer_stats(expert_counts=counts)


@pytest.fixture(scope="function")
def layer_stats_empty() -> LayerStats:
    """Empty LayerStats (no events yet)."""
    return LayerStats(
        layer_name="moe_layers.empty",
        n_experts=0,
        expert_counts=torch.zeros(0, dtype=torch.int64),
        total_tokens=0,
        utilization=torch.zeros(0, dtype=torch.float32),
        load_imbalance_score=float("nan"),
        event_count=0,
        step_range=(-1, -1),
    )


# =============================================================================
# §7  STAT COLLECTOR FACTORY
# =============================================================================

@pytest.fixture(scope="function")
def seeded_collector(default_config) -> StatCollector:
    """StatCollector pre-seeded with 20 uniform events on two layers."""
    layer_names = ["moe_layers.0", "moe_layers.1"]
    collector   = StatCollector(layer_names=layer_names, config=default_config)

    for step in range(20):
        for layer_name in layer_names:
            event = _make_routing_event(
                layer_name=layer_name,
                n_experts=8,
                step=step,
                expert_counts=[100] * 8,
            )
            collector.add_event(event)

    return collector


@pytest.fixture(scope="function")
def collapsed_seeded_collector(default_config) -> StatCollector:
    """StatCollector seeded with collapse events on layer 0 (expert 0 = 100%)."""
    layer_names = ["moe_layers.0"]
    collector   = StatCollector(layer_names=layer_names, config=default_config)

    for step in range(10):
        event = _make_routing_event(
            layer_name="moe_layers.0",
            n_experts=8,
            step=step,
            expert_counts=[800, 0, 0, 0, 0, 0, 0, 0],
        )
        collector.add_event(event)

    return collector


# =============================================================================
# §8  UTILITY HELPERS (exported for use in test modules)
# =============================================================================

def make_routing_event(
    layer_name: str = "moe_layers.0",
    n_experts: int = 8,
    step: int = 1,
    expert_counts: Optional[List[int]] = None,
    raw_logits: Optional[torch.Tensor] = None,
) -> RoutingEvent:
    """Module-level alias of _make_routing_event for import in test files."""
    return _make_routing_event(
        layer_name=layer_name,
        n_experts=n_experts,
        step=step,
        expert_counts=expert_counts,
        raw_logits=raw_logits,
    )


def make_layer_stats(
    layer_name: str = "moe_layers.0",
    n_experts: int = 8,
    expert_counts: Optional[List[int]] = None,
    has_raw_logits: bool = False,
    raw_logits_window: Optional[list] = None,
    event_count: int = 10,
) -> LayerStats:
    """Module-level alias of _make_layer_stats for import in test files."""
    return _make_layer_stats(
        layer_name=layer_name,
        n_experts=n_experts,
        expert_counts=expert_counts,
        has_raw_logits=has_raw_logits,
        raw_logits_window=raw_logits_window,
        event_count=event_count,
    )


def run_forward_passes(
    model: nn.Module,
    n_passes: int = 5,
    batch_size: int = 2,
    seq_len: int = 16,
    vocab_size: int = 256,
) -> None:
    """Run *n_passes* forward passes on *model* with random inputs."""
    with torch.no_grad():
        for _ in range(n_passes):
            ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            model(ids)
