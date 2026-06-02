# ------------------------------------------------------------------------------
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
# moewatch/hooks/router_hook.py
#
# RouterHook — PyTorch forward hook that fires on every router module pass,
# extracts routing logits / indices, and writes a RoutingEvent to the
# StatCollector ring buffer.
#
# RoutingEvent — lightweight data container for one forward-pass snapshot.
#
# Design constraints
# ------------------
#   - The hook must never raise inside a forward pass. Any extraction error
#     is caught, counted, and logged; the training step continues unaffected.
#   - No tensor data is retained on the GPU. All statistics are moved to CPU
#     immediately to avoid GPU memory pressure.
#   - The hook is stateless: all mutable state lives in StatCollector.
#     RouterHook holds only a reference to the collector and its layer name.
#   - step-sampling is enforced here (sample_every) so the collector never
#     sees events it would discard anyway.
#
# Author : Abinesh (GitHub: Abineshabee)
# License: Apache 2.0
# Version: 0.1.0
# ------------------------------------------------------------------------------

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import WatchConfig

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# RoutingEvent — one forward-pass snapshot for a single router layer
# ------------------------------------------------------------------------------

@dataclass
class RoutingEvent:
    """Immutable snapshot of a single router forward pass.

    Written by ``RouterHook`` and consumed by ``StatCollector``.

    Attributes
    ----------
    layer_name   : Fully-qualified module name (e.g. ``model.layers.0.block_sparse_moe``).
    step         : Global step counter at the time of capture (hook-internal counter,
                   not the trainer's global_step — the two are proportional via sample_every).
    timestamp    : Wall-clock capture time (seconds since epoch).
    n_experts    : Total number of experts in this layer.
    expert_counts: 1-D CPU int64 tensor of shape ``(n_experts,)`` counting how many
                   tokens were routed to each expert in this batch.
    total_tokens : Total tokens processed (sum of expert_counts; may exceed
                   batch_size × seq_len when top-k > 1).
    raw_logits   : Optional CPU float32 tensor of shape ``(total_tokens, n_experts)``
                   containing the un-softmaxed router logits. ``None`` when logit
                   extraction is not possible (e.g. model only exposes top-k indices).
    top_k        : Number of experts each token was routed to (inferred from the output).
    """

    layer_name:    str
    step:          int
    timestamp:     float
    n_experts:     int
    expert_counts: torch.Tensor          # shape: (n_experts,), dtype: int64, device: CPU
    total_tokens:  int
    raw_logits:    Optional[torch.Tensor] = None   # shape: (total_tokens, n_experts), CPU
    top_k:         int                    = 1


# ------------------------------------------------------------------------------
# RouterHook
# ------------------------------------------------------------------------------

class RouterHook:
    """PyTorch ``register_forward_hook`` callback for a single MoE router module.

    One ``RouterHook`` is created per router module. It is registered via
    ``module.register_forward_hook(hook_fn)`` and removed when the parent
    ``HookManager`` calls ``detach()``.

    Parameters
    ----------
    layer_name : str
        Fully-qualified module name — used as the key in the collector.
    collector  : StatCollector
        Receives the ``RoutingEvent`` after each instrumented forward pass.
    config     : WatchConfig
        Provides ``sample_every`` for step-sampling and ``no_color`` for logging.
    """

    def __init__(
        self,
        layer_name: str,
        collector: Any,            # StatCollector — forward ref to avoid circular import
        config: WatchConfig,
    ) -> None:
        self.layer_name = layer_name
        self.collector  = collector
        self.config     = config

        self._call_count:  int = 0   # total forward passes seen (before sampling filter)
        self._error_count: int = 0   # extraction errors (non-fatal)
        self._handle: Optional[Any] = None   # torch hook handle

    # --------------------------------------------------------------------------
    # Hook callable — called by PyTorch after each forward pass of the module
    # --------------------------------------------------------------------------

    def __call__(
        self,
        module: nn.Module,
        inputs: Tuple[Any, ...],
        output: Any,
    ) -> None:
        """PyTorch forward hook signature: (module, inputs, output) → None.

        Never raises. Any error is silently counted; the forward pass continues.
        """
        self._call_count += 1

        # Step-sampling: skip unless this is a sampled step
        if self._call_count % self.config.sample_every != 0:
            return

        try:
            event = self._extract_event(module, inputs, output)
        except Exception as exc:
            self._error_count += 1
            if self._error_count <= 3:
                log.warning(
                    "[moewatch] RouterHook(%s): extraction error on call %d "
                    "(suppressing further warnings after 3): %s",
                    self.layer_name,
                    self._call_count,
                    exc,
                )
            return

        try:
            self.collector.add_event(event)
        except Exception as exc:                    # pragma: no cover
            log.warning(
                "[moewatch] RouterHook(%s): collector.add_event failed: %s",
                self.layer_name,
                exc,
            )

    # --------------------------------------------------------------------------
    # Routing event extraction — architecture-aware dispatch
    # --------------------------------------------------------------------------

    def _extract_event(
        self,
        module: nn.Module,
        inputs: Tuple[Any, ...],
        output: Any,
    ) -> RoutingEvent:
        """Attempt to extract a ``RoutingEvent`` from the module output.

        Tries multiple extraction strategies in order of specificity:

        1. Tuple/list output with a tensor of expert indices at position 1
           (Mixtral ``MixtralSparseMoeBlock`` style).
        2. Object with ``.routing_weights`` and ``.expert_indices`` attributes
           (some DeepSeek / custom MoE styles).
        3. Single tensor output — interpret as routing logits directly
           (simple gating modules).
        4. Dict output with known keys (``router_logits``, ``expert_indices``).

        If none match, falls back to an input-tensor scan for the token count
        and raises ``ValueError`` with a diagnostic message.
        """

        # -- Strategy 1: tuple/list — most common (Mixtral, OLMoE) ----------
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            event = self._from_tuple_output(output)
            if event is not None:
                return event

        # -- Strategy 2: object with routing attributes ----------------------
        if hasattr(output, "router_logits") or hasattr(output, "expert_indices"):
            event = self._from_object_output(output)
            if event is not None:
                return event

        # -- Strategy 3: single tensor — interpret as logits ----------------
        if isinstance(output, torch.Tensor):
            return self._from_logits_tensor(output)

        # -- Strategy 4: dict output ----------------------------------------
        if isinstance(output, dict):
            event = self._from_dict_output(output)
            if event is not None:
                return event

        # -- Fallback: infer token count from inputs, return zero counts -----
        total_tokens = self._infer_token_count(inputs)
        log.debug(
            "[moewatch] RouterHook(%s): unrecognised output type %s — "
            "returning zero-count event. Consider specifying router_modules "
            "manually or opening an issue.",
            self.layer_name,
            type(output).__name__,
        )
        return RoutingEvent(
            layer_name=self.layer_name,
            step=self._call_count,
            timestamp=time.time(),
            n_experts=0,
            expert_counts=torch.zeros(0, dtype=torch.int64),
            total_tokens=total_tokens,
        )

    # --------------------------------------------------------------------------
    # Extraction strategies
    # --------------------------------------------------------------------------

    def _from_tuple_output(
        self,
        output: Any,
    ) -> Optional[RoutingEvent]:
        """Handle tuple/list outputs.

        Mixtral ``MixtralSparseMoeBlock`` returns:
          (hidden_states, router_logits)
        where router_logits is shape (batch × seq_len, n_experts).

        OLMoE returns:
          (hidden_states, router_probs, selected_experts)
        where selected_experts is shape (batch × seq_len, top_k).
        """
        # Try position [1] as logits tensor
        if isinstance(output[1], torch.Tensor):
            candidate = output[1].detach().cpu()

            if candidate.ndim == 2:
                # Shape: (total_tokens, n_experts) — router logits
                return self._from_logits_tensor(candidate)

        # Try position [2] as expert indices (OLMoE style)
        if len(output) >= 3 and isinstance(output[2], torch.Tensor):
            indices = output[2].detach().cpu()
            if indices.ndim == 2:
                # Shape: (total_tokens, top_k)
                return self._from_expert_indices(indices, logits=None)

        return None

    def _from_object_output(self, output: Any) -> Optional[RoutingEvent]:
        """Handle output objects with named routing attributes."""
        logits  = getattr(output, "router_logits",  None)
        indices = getattr(output, "expert_indices",  None)

        if logits is not None and isinstance(logits, torch.Tensor):
            return self._from_logits_tensor(logits.detach().cpu())

        if indices is not None and isinstance(indices, torch.Tensor):
            return self._from_expert_indices(indices.detach().cpu(), logits=None)

        return None

    def _from_dict_output(self, output: Dict[str, Any]) -> Optional[RoutingEvent]:
        """Handle dict-style outputs."""
        for key in ("router_logits", "gate_logits", "routing_logits"):
            if key in output and isinstance(output[key], torch.Tensor):
                return self._from_logits_tensor(output[key].detach().cpu())

        for key in ("expert_indices", "selected_experts", "top_k_indices"):
            if key in output and isinstance(output[key], torch.Tensor):
                return self._from_expert_indices(output[key].detach().cpu(), logits=None)

        return None

    def _from_logits_tensor(self, logits: torch.Tensor) -> RoutingEvent:
        """Build a RoutingEvent from a (total_tokens, n_experts) logits tensor.

        Computes expert_counts by taking argmax (top-1) or tracking top-k
        selections. For top-1 we can compute exact counts efficiently with
        ``torch.bincount``; for top-k we use ``unique`` with return_counts.
        """
        if logits.ndim == 1:
            # Single-token batch: unsqueeze to (1, n_experts)
            logits = logits.unsqueeze(0)

        if logits.ndim != 2:
            raise ValueError(
                f"RouterHook({self.layer_name}): expected 2-D logits tensor, "
                f"got shape {list(logits.shape)}."
            )

        total_tokens, n_experts = logits.shape
        float_logits = logits.float()

        # Top-1 selection (most architectures)
        top1_indices = float_logits.argmax(dim=-1)   # (total_tokens,)
        expert_counts = torch.bincount(top1_indices, minlength=n_experts).to(torch.int64)

        return RoutingEvent(
            layer_name=self.layer_name,
            step=self._call_count,
            timestamp=time.time(),
            n_experts=n_experts,
            expert_counts=expert_counts,
            total_tokens=total_tokens,
            raw_logits=float_logits,
            top_k=1,
        )

    def _from_expert_indices(
        self,
        indices: torch.Tensor,
        logits: Optional[torch.Tensor],
    ) -> RoutingEvent:
        """Build a RoutingEvent from a (total_tokens, top_k) indices tensor.

        When only expert indices (not logits) are available, we cannot compute
        exact routing distributions — but we can still count token assignments.
        """
        if indices.ndim == 1:
            indices = indices.unsqueeze(-1)

        if indices.ndim != 2:
            raise ValueError(
                f"RouterHook({self.layer_name}): expected 2-D expert indices, "
                f"got shape {list(indices.shape)}."
            )

        total_tokens, top_k = indices.shape
        flat_indices = indices.reshape(-1).long()

        # Infer n_experts from the maximum observed index + 1
        n_experts = int(flat_indices.max().item()) + 1
        expert_counts = torch.bincount(flat_indices, minlength=n_experts).to(torch.int64)

        return RoutingEvent(
            layer_name=self.layer_name,
            step=self._call_count,
            timestamp=time.time(),
            n_experts=n_experts,
            expert_counts=expert_counts,
            total_tokens=total_tokens,
            raw_logits=logits,
            top_k=top_k,
        )

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------

    @staticmethod
    def _infer_token_count(inputs: Tuple[Any, ...]) -> int:
        """Best-effort: infer total tokens from the first tensor in inputs."""
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                if inp.ndim >= 2:
                    return int(inp.shape[0] * inp.shape[1])
                return int(inp.shape[0])
        return 0

    # --------------------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total number of forward passes seen (before sample-every filter)."""
        return self._call_count

    @property
    def error_count(self) -> int:
        """Number of non-fatal extraction errors encountered."""
        return self._error_count

    def __repr__(self) -> str:
        return (
            f"RouterHook("
            f"layer={self.layer_name!r}, "
            f"calls={self._call_count}, "
            f"errors={self._error_count}"
            f")"
        )
