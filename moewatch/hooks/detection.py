# ------------------------------------------------------------------------------
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
# moewatch/hooks/detection.py
#
# detect_router_modules() — auto-detect MoE router modules in any HuggingFace
# model via a two-pass strategy:
#
#   Pass 1 — Architecture registry
#     Match against a curated registry of known class names for Mixtral,
#     OLMoE, DeepSeek-MoE, Qwen-MoE, and Switch Transformer. O(n_modules)
#     single scan; returns immediately on a registry hit.
#
#   Pass 2 — Heuristic fallback
#     Scan all module class names for substrings associated with MoE routing:
#     "router", "gate", "sparse", "moe", "expert". Filters out obvious
#     non-router classes (embedding, norm, attention) to reduce false positives.
#
# Both passes return fully-qualified module names (as returned by
# model.named_modules()) that can be passed directly to HookManager.
#
# Graceful degradation
# --------------------
# If neither pass finds anything, the function returns an empty list and
# logs a detailed diagnostic rather than raising. The caller (_audit.py /
# _watcher.py) is responsible for deciding whether an empty result is fatal.
#
# Extending the registry
# ----------------------
# Add a new architecture by appending its router class name(s) to
# _ARCHITECTURE_REGISTRY below. No other changes are required.
#
# Author : Abinesh (GitHub: Abineshabee)
# License: Apache 2.0
# Version: 0.1.0
# ------------------------------------------------------------------------------

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Set

import torch.nn as nn

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Architecture registry
#
# Maps architecture family name → frozenset of known router class names.
# Class names are matched against type(module).__name__ (exact match, case-
# sensitive). This is intentionally more specific than substring matching to
# avoid false positives on embedding or attention layers that happen to contain
# "gate" or "expert" in the class name.
# ------------------------------------------------------------------------------

_ARCHITECTURE_REGISTRY: Dict[str, FrozenSet[str]] = {
    # Mixtral — meta-llama / mistralai
    "Mixtral": frozenset(
        {
            "MixtralSparseMoeBlock",
            "MixtralBlocSparseTop2MLP",  # some forks rename the block
            "MixtralTopKRouter",          # newer transformers: standalone router gate
        }
    ),
    # OLMoE — allenai
    "OLMoE": frozenset(
        {
            "OlmoeMoE",
            "OlmoeSparseMoeBlock",
        }
    ),
    # DeepSeek-MoE — deepseek-ai
    "DeepSeek": frozenset(
        {
            "DeepseekMoE",
            "DeepseekV2MoE",
            "DeepseekV3MoE",
            "MoEGate",  # standalone gate module in DS-V2/V3
        }
    ),
    # Qwen-MoE — Alibaba
    "Qwen": frozenset(
        {
            "QwenMoE",
            "Qwen2MoeSparseMoeBlock",
            "Qwen3MoeSparseMoeBlock",
        }
    ),
    # Switch Transformer — Google (HuggingFace port)
    "SwitchTransformer": frozenset(
        {
            "SwitchTransformersSparseMLP",
            "SwitchTransformersTop1Router",
        }
    ),
    # Phi-MoE — Microsoft
    "Phi": frozenset(
        {
            "PhiMoE",
            "PhiMoESparseMoeBlock",
        }
    ),
    # NLLB-MoE / mBART-MoE — Meta
    "NllbMoE": frozenset(
        {
            "NllbMoeSparseMLP",
            "NllbMoeTop2Router",
        }
    ),
    # Arctic — Snowflake
    "Arctic": frozenset(
        {
            "ArcticMoE",
            "ArcticMoeBlock",
        }
    ),
    # Jamba — AI21 Labs
    "Jamba": frozenset(
        {
            "JambaMoE",
            "JambaSparseMoeBlock",
        }
    ),
}

# Flattened set for O(1) membership tests
_ALL_KNOWN_CLASSES: FrozenSet[str] = frozenset(
    cls for classes in _ARCHITECTURE_REGISTRY.values() for cls in classes
)

# ------------------------------------------------------------------------------
# Heuristic fallback configuration
# ------------------------------------------------------------------------------

# Substrings in class name that suggest a router/gate module
_ROUTER_SUBSTRINGS: FrozenSet[str] = frozenset(
    {
        "router",
        "gate",
        "sparse",
        "moe",
        "expert_select",
        "top_k",
        "topk",
        "routing",
    }
)

# Substrings that disqualify a module even if it matches a router substring
# (e.g. "GateProjection" in LLaMA FFN is not an MoE router)
_EXCLUSION_SUBSTRINGS: FrozenSet[str] = frozenset(
    {
        "embedding",
        "embed",
        "norm",
        "ln_",
        "layernorm",
        "attention",
        "attn",
        "mlp",  # plain FFN MLP — not a router
        "projection",
        "proj",
        "lm_head",
        "head",
        "dropout",
        "act",
        "activation",
    }
)

# Minimum number of parameters a module must have to be considered a router
# (rules out tiny bias-only or scalar modules whose name contains "gate")
_MIN_PARAM_COUNT: int = 1


def _remove_ancestor_duplicates(names: List[str]) -> List[str]:
    names = sorted(names, key=len, reverse=True)  # deepest first
    result: List[str] = []

    for name in names:
        if not any(prev.startswith(name + ".") for prev in result):
            result.append(name)

    return sorted(result)


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


def detect_router_modules(model: nn.Module) -> List[str]:
    """Auto-detect MoE router module names in *model*.

    Returns a list of fully-qualified module names (as produced by
    ``model.named_modules()``) that correspond to MoE router / gate layers.

    The list is deduplicated and ordered by depth (shallowest first), which
    matches the natural layer ordering in transformer models.

    Parameters
    ----------
    model : torch.nn.Module
        Any HuggingFace MoE model. Must have been fully initialised (weights
        loaded) before calling this function so that parameter counts are
        meaningful.

    Returns
    -------
    list of str
        Fully-qualified router module names. Empty list if detection fails
        (a detailed warning is logged in that case).

    Notes
    -----
    The returned names can be passed directly to ``WatchConfig(router_modules=...)``
    to bypass auto-detection on subsequent runs — useful for reproducibility.
    """

    found_registry: List[str] = []
    found_heuristic: List[str] = []
    seen: Set[str] = set()

    # Count total modules for diagnostic logging
    all_modules = list(model.named_modules())
    total_count = len(all_modules)

    log.debug(
        "[moewatch] detect_router_modules: scanning %d modules ...",
        total_count,
    )

    for name, module in all_modules:
        if not name:
            # Skip the root module itself (name == "")
            continue

        class_name = type(module).__name__

        # -- Pass 1: registry hit -------------------------------------------
        if class_name in _ALL_KNOWN_CLASSES:
            if name not in seen:
                found_registry.append(name)
                seen.add(name)
                log.debug(
                    "[moewatch] Registry hit: %s (%s)",
                    name,
                    class_name,
                )
            continue  # no need to also check heuristic

        # -- Pass 2: heuristic fallback -------------------------------------
        class_lower = class_name.lower()

        has_router_substring = any(sub in class_lower for sub in _ROUTER_SUBSTRINGS)
        has_exclusion = any(ex in class_lower for ex in _EXCLUSION_SUBSTRINGS)

        if has_router_substring and not has_exclusion:
            # Final guard: must have at least one learnable parameter
            param_count = sum(1 for _ in module.parameters(recurse=True))
            if param_count >= _MIN_PARAM_COUNT:
                if name not in seen:
                    found_heuristic.append(name)
                    seen.add(name)
                    log.debug(
                        "[moewatch] Heuristic hit: %s (%s)",
                        name,
                        class_name,
                    )

    # --------------------------------------------------------------------------
    # Merge and report
    # --------------------------------------------------------------------------

    # Registry hits always win; heuristic results are only used when registry
    # finds nothing at all. This prevents false-positive heuristic hits from
    # polluting a clean registry detection on known architectures.
    result: List[str] = []

    if found_registry:
        result = _remove_ancestor_duplicates(found_registry)
        log.info(
            "[moewatch] Architecture registry detected %d router module(s) "
            "(heuristic found %d additional — ignored because registry succeeded).",
            len(result),
            len(found_heuristic),
        )
        _log_detected(result)
        return result

    if found_heuristic:
        result = _remove_ancestor_duplicates(found_heuristic)
        log.info(
            "[moewatch] Heuristic detection found %d router module(s). "
            "If these look wrong, use WatchConfig(router_modules=[...]) "
            "to specify them explicitly.",
            len(result),
        )
        _log_detected(result)
        return result

    # --------------------------------------------------------------------------
    # Nothing found — emit a detailed diagnostic
    # --------------------------------------------------------------------------
    _log_detection_failure(model, total_count)
    return []


# ------------------------------------------------------------------------------
# Registry introspection helpers (useful for contributors and debugging)
# ------------------------------------------------------------------------------


def list_known_architectures() -> Dict[str, List[str]]:
    """Return the full architecture registry as a plain dict of lists.

    Useful for documentation generation and ``--list-architectures`` CLI flags.
    """
    return {
        family: sorted(classes) for family, classes in _ARCHITECTURE_REGISTRY.items()
    }


def is_known_router_class(class_name: str) -> bool:
    """Return True if *class_name* is in the curated architecture registry."""
    return class_name in _ALL_KNOWN_CLASSES


# ------------------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------------------


def _log_detected(names: List[str]) -> None:
    """Log up to 5 detected names at INFO level."""
    preview = names[:5]
    suffix = f" (+ {len(names) - 5} more)" if len(names) > 5 else ""
    log.info(
        "[moewatch] Detected router modules:\n  %s%s",
        "\n  ".join(preview),
        suffix,
    )


def _log_detection_failure(model: nn.Module, total_count: int) -> None:
    """Emit a detailed warning when detection finds nothing."""
    # Sample up to 20 module class names to help the user diagnose manually
    sample_classes = list(
        {type(m).__name__ for _, m in list(model.named_modules())[:50]}
    )[:20]

    log.warning(
        "[moewatch] Router module detection found nothing in a model with "
        "%d modules.\n\n"
        "  Scanned class names (sample):\n    %s\n\n"
        "  This usually means:\n"
        "    1. The model uses a non-standard MoE class name not in the registry.\n"
        "    2. The model is a dense (non-MoE) model.\n"
        "    3. The model was loaded as a shell without weights.\n\n"
        "  To fix:\n"
        "    • List all modules: for name, m in model.named_modules(): print(name, type(m).__name__)\n"
        "    • Then pass the correct names: WatchConfig(router_modules=['...'])\n"
        "    • Or open a GitHub issue to add your architecture to the registry:\n"
        "      https://github.com/Abineshabee/moewatch/issues",
        total_count,
        "\n    ".join(sample_classes),
    )
