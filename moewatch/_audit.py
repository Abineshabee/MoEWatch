# =============================================================================
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
#  _audit.py — Offline diagnostic entry point: audit(model, dataloader, ...)
#
#  ``audit()`` is the primary API for researchers who want a one-shot
#  diagnostic without modifying their training loop. It:
#
#    1. Auto-detects (or accepts manual) router modules via HookManager.
#    2. Runs N forward passes through the model, sampling routing statistics.
#    3. Analyzes collected stats (entropy, collapse, imbalance).
#    4. Returns a structured AuditReport — no side effects, no mutation.
#
#  Design principles:
#    - Context-manager lifecycle: hooks are always cleaned up, even on crash.
#    - No gradient interference: runs under torch.no_grad() by default.
#    - Device-agnostic: works on CPU, single GPU, and multi-GPU inference.
#    - Graceful degradation: unknown architectures warn and return empty report.
#
#  Author : Abinesh (GitHub: Abineshabee)
#  License: Apache 2.0
#  Version: 0.1.0
#
# =============================================================================

from __future__ import annotations

import logging
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterable, Iterator, Optional
from .report.audit_report import AuditReport

import torch
import torch.nn as nn

from .config import OutputMode, WatchConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — keep top-level import fast; sub-packages are heavy
# ---------------------------------------------------------------------------


def _import_hooks():
    from .hooks.manager import HookManager
    from .hooks.detection import detect_router_modules

    return HookManager, detect_router_modules


def _import_collector():
    from .collector.stat_collector import StatCollector

    return StatCollector


def _import_analyzers():
    from .analyzer.entropy import EntropyAnalyzer
    from .analyzer.collapse import CollapseDetector

    return EntropyAnalyzer, CollapseDetector


def _import_report():
    from .report.audit_report import AuditReport
    from .report.cli_reporter import CLIReporter

    return AuditReport, CLIReporter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_device(model: nn.Module) -> torch.device:
    """Return the device of the first parameter in *model*.

    Falls back to CPU if the model has no parameters (e.g. a wrapper shell
    loaded before weights are attached).
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _is_dataloader_empty(dataloader: Any) -> bool:
    """Best-effort emptiness check without consuming the iterator."""
    try:
        return len(dataloader) == 0  # type: ignore[arg-type]
    except TypeError:
        # IterableDataset — cannot know length without consuming
        return False


def _make_synthetic_batch(
    model: nn.Module,
    device: torch.device,
    seq_len: int = 32,
    batch_size: int = 1,
) -> Dict[str, torch.Tensor]:
    """Create a minimal random input batch for models with no dataloader.

    Attempts to infer ``vocab_size`` from the model's embedding layer.
    Falls back to 32_000 (Mixtral / LLaMA default) if not found.
    """
    vocab_size = 32_000
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            vocab_size = module.num_embeddings
            break

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@contextmanager
def _no_grad_ctx(use_no_grad: bool) -> Generator[None, None, None]:
    """Conditionally wrap execution in ``torch.no_grad()``."""
    if use_no_grad:
        with torch.no_grad():
            yield
    else:
        yield


def _forward_pass(
    model: nn.Module,
    batch: Any,
    device: torch.device,
) -> None:
    """Execute a single forward pass, moving batch tensors to *device* if needed.

    Returns nothing — we only care about the side-effects in the hooks.
    Silences all model outputs to avoid any accidental gradient accumulation.
    """
    # Move dict-style batches to the target device
    if isinstance(batch, dict):
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        model(**batch)

    elif isinstance(batch, (list, tuple)):
        # (input_ids, attention_mask, ...) positional style
        batch = tuple(v.to(device) if isinstance(v, torch.Tensor) else v for v in batch)
        model(*batch)

    elif isinstance(batch, torch.Tensor):
        model(batch.to(device))

    else:
        # Unknown batch type — try calling model directly and warn
        warnings.warn(
            f"[moewatch] Unrecognised batch type {type(batch).__name__}. "
            "Passing it to model() directly. Override batch handling via a "
            "custom dataloader that returns dicts.",
            UserWarning,
            stacklevel=4,
        )
        model(batch)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit(
    model: nn.Module,
    dataloader: Optional[Iterable[Any]] = None,
    *,
    steps: int = 50,
    config: Optional[WatchConfig] = None,
    use_no_grad: bool = True,
    verbose: bool = True,
) -> "AuditReport":  # noqa: F821  (resolved lazily)
    """Run a full offline diagnostic pass on a HuggingFace MoE model.

    This is the primary entry point for researchers who want to audit a model
    *without* modifying their training loop. Pass the model and an optional
    dataloader; ``audit()`` instruments the router modules via forward hooks,
    runs *steps* forward passes, collects routing statistics, and returns a
    structured ``AuditReport``.

    Parameters
    ----------
    model : torch.nn.Module
        Any HuggingFace MoE model loaded with ``AutoModelForCausalLM`` or
        equivalent. The model is **never modified** — ``audit()`` attaches
        and detaches hooks without touching weights or buffers.

    dataloader : iterable of batches, optional
        A standard PyTorch ``DataLoader`` or any iterable yielding batches
        (dict, tuple, or raw tensor). If ``None``, moewatch synthesises a
        random input batch from the model's embedding vocabulary — useful for
        quick sanity checks when you do not have data at hand.

    steps : int
        Number of forward passes to run for diagnostic sampling. More steps
        give a more representative picture of routing behaviour but take longer.
        Default ``50``. For quick sanity checks use ``steps=10``; for
        publication-quality audits use ``steps=200``.

    config : WatchConfig, optional
        Full diagnostic configuration. If ``None``, ``WatchConfig()`` defaults
        are used. Use ``WatchConfig.aggressive()`` for debugging collapse or
        ``WatchConfig.lightweight()`` for large-scale models.

    use_no_grad : bool
        When ``True`` (default), wraps all forward passes in
        ``torch.no_grad()`` to prevent gradient accumulation and reduce memory
        overhead. Set to ``False`` only if you are auditing inside a training
        step and need gradients to flow.

    verbose : bool
        Print a brief header and progress to stdout while the audit runs.
        Default ``True``. Set to ``False`` for programmatic use or when
        ``config.output == OutputMode.SILENT``.

    Returns
    -------
    AuditReport
        Structured diagnostic result. Key methods:

        - ``report.summary()``         — full human-readable report
        - ``report.dead_experts()``    — per-layer list of dead/cold experts
        - ``report.routing_entropy()`` — per-layer entropy scores
        - ``report.utilization()``     — per-expert token distribution
        - ``report.to_json()``         — machine-readable JSON export

    Raises
    ------
    TypeError
        If *model* is not a ``torch.nn.Module``.
    ValueError
        If *steps* < 1.
    RuntimeError
        If no router modules could be detected and no manual override was
        provided via ``config.router_modules``.

    Warns
    -----
    UserWarning
        If fewer router modules are detected than expected, or if the
        dataloader is exhausted before *steps* forward passes complete.

    Examples
    --------
    Minimal usage (synthetic data):

    >>> from moewatch import audit
    >>> report = audit(model)
    >>> report.summary()

    With a real DataLoader:

    >>> report = audit(model, train_dataloader, steps=100)
    >>> if report.dead_experts():
    ...     print("Expert collapse detected!")

    Silent mode for CI integration:

    >>> from moewatch import audit, WatchConfig
    >>> report = audit(model, config=WatchConfig.silent())
    >>> dead = report.dead_experts()
    """

    # ------------------------------------------------------------------
    # 0. Input validation
    # ------------------------------------------------------------------
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"[moewatch] audit() expects a torch.nn.Module, "
            f"got {type(model).__name__}."
        )

    if not isinstance(steps, int) or steps < 1:
        raise ValueError(f"[moewatch] steps must be a positive integer, got {steps!r}.")

    if config is None:
        config = WatchConfig()

    # ------------------------------------------------------------------
    # 1. Resolve device and prepare data iterator
    # ------------------------------------------------------------------
    device = _resolve_device(model)

    if dataloader is None:
        warnings.warn(
            "[moewatch] No dataloader provided. Using synthetic random "
            "batches for diagnostic sampling. Results may not reflect "
            "real-world routing behaviour. Pass a dataloader for accurate "
            "diagnostics.",
            UserWarning,
            stacklevel=2,
        )
        synthetic_batch = _make_synthetic_batch(model, device)
        data_iter: Iterator[Any] = _infinite_iter(synthetic_batch)
        dataloader_exhausted_ok = True
    else:
        if _is_dataloader_empty(dataloader):
            raise ValueError(
                "[moewatch] The provided dataloader appears to be empty. "
                "Ensure it yields at least one batch."
            )
        data_iter = iter(dataloader)
        dataloader_exhausted_ok = False

    # ------------------------------------------------------------------
    # 2. Lazy-import sub-modules
    # ------------------------------------------------------------------
    HookManager, detect_router_modules = _import_hooks()
    StatCollector = _import_collector()
    EntropyAnalyzer, CollapseDetector = _import_analyzers()
    AuditReport, CLIReporter = _import_report()

    # ------------------------------------------------------------------
    # 3. Detect (or accept manual) router modules
    # ------------------------------------------------------------------
    if config.router_modules:
        router_module_names = list(config.router_modules)
        log.debug(
            "[moewatch] Using %d manually specified router module(s).",
            len(router_module_names),
        )
    else:
        router_module_names = detect_router_modules(model)
        if not router_module_names:
            raise RuntimeError(
                "[moewatch] Could not detect any router modules in the model. "
                "moewatch supports Mixtral, OLMoE, DeepSeek-MoE, and Qwen-MoE "
                "architectures out of the box. For custom MoE architectures, "
                "specify router_modules explicitly via WatchConfig:\n\n"
                "  config = WatchConfig(\n"
                '      router_modules=["model.layers.0.mlp.gate"]\n'
                "  )\n\n"
                "To list all named modules in your model:\n"
                "  for name, _ in model.named_modules(): print(name)"
            )

    # ------------------------------------------------------------------
    # 4. Set up collector and hook manager
    # ------------------------------------------------------------------
    collector = StatCollector(
        layer_names=router_module_names,
        config=config,
    )

    hook_manager = HookManager(
        model=model,
        router_module_names=router_module_names,
        collector=collector,
        config=config,
    )

    # ------------------------------------------------------------------
    # 5. Header
    # ------------------------------------------------------------------
    effective_verbose = verbose and config.output != OutputMode.SILENT

    if effective_verbose:
        _print_audit_header(
            n_routers=len(router_module_names),
            steps=steps,
            device=device,
            config=config,
        )

    # ------------------------------------------------------------------
    # 6. Main sampling loop — hooks attached for duration, always torn down
    # ------------------------------------------------------------------
    completed_steps = 0
    start_time = time.perf_counter()

    with hook_manager:
        with _no_grad_ctx(use_no_grad):
            for step in range(steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if dataloader_exhausted_ok:
                        # Should not happen with infinite synthetic iter
                        break
                    warnings.warn(
                        f"[moewatch] Dataloader exhausted after {completed_steps} "
                        f"steps (requested {steps}). Audit will proceed with "
                        "the data collected so far. For a full audit, use a "
                        "larger dataset or reduce steps.",
                        UserWarning,
                        stacklevel=2,
                    )
                    break

                _forward_pass(model, batch, device)
                completed_steps += 1

                if effective_verbose and (step + 1) % max(1, steps // 10) == 0:
                    pct = 100 * (step + 1) / steps
                    print(
                        f"  [moewatch] Sampling ... {step + 1}/{steps} "
                        f"({pct:.0f}%)",
                        flush=True,
                    )

    elapsed = time.perf_counter() - start_time

    # ------------------------------------------------------------------
    # 7. Warn if too few steps were completed for reliable analysis
    # ------------------------------------------------------------------
    if completed_steps < 10:
        warnings.warn(
            f"[moewatch] Only {completed_steps} forward pass(es) completed. "
            "Diagnostic results may be unreliable with fewer than 10 steps. "
            "Increase the dataloader size or reduce 'steps'.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 8. Analyze collected statistics
    # ------------------------------------------------------------------
    layer_stats = collector.get_all_stats()
    entropy_analyzer = EntropyAnalyzer(config=config)
    collapse_detector = CollapseDetector(config=config)

    entropy_results = entropy_analyzer.analyze(layer_stats)
    collapse_results = collapse_detector.detect(layer_stats)

    # ------------------------------------------------------------------
    # 9. Build and return report
    # ------------------------------------------------------------------
    report = AuditReport(
        layer_stats=layer_stats,
        entropy_results=entropy_results,
        collapse_results=collapse_results,
        config=config,
        completed_steps=completed_steps,
        elapsed_seconds=elapsed,
        device=str(device),
        router_module_names=router_module_names,
    )

    if effective_verbose:
        reporter = CLIReporter(config=config)
        reporter.print_summary(report)

    return report


# ---------------------------------------------------------------------------
# Private iterator helper
# ---------------------------------------------------------------------------


def _infinite_iter(batch: Any) -> Iterator[Any]:
    """Yield *batch* indefinitely (for synthetic-data mode)."""
    while True:
        yield batch


# ---------------------------------------------------------------------------
# Pretty header
# ---------------------------------------------------------------------------


def _print_audit_header(
    n_routers: int,
    steps: int,
    device: torch.device,
    config: WatchConfig,
) -> None:
    """Print a structured ASCII header at the start of an audit run."""

    _LINE = "─" * 66

    print()
    print(f"  ╔{_LINE}╗")
    print(f"  ║{'moewatch · Offline Diagnostic Audit':^66}║")
    print(f"  ╠{_LINE}╣")
    print(f"  ║  {'Router modules detected':.<40} {n_routers:>8}        ║")
    print(f"  ║  {'Forward passes (steps)':.<40} {steps:>8}        ║")
    print(f"  ║  {'Sample every N steps':.<40} {config.sample_every:>8}        ║")
    print(f"  ║  {'Device':.<40} {str(device):>8}        ║")
    print(f"  ║  {'Dead expert threshold':.<40} {config.dead_threshold:>7.3%}        ║")
    print(f"  ║  {'Entropy warn threshold':.<40} {config.entropy_warn:>7.1%}        ║")
    print(f"  ╚{_LINE}╝")
    print()
