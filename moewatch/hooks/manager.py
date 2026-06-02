# ------------------------------------------------------------------------------
#
#  ███╗   ███╗ ██████╗ ███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
#  ████╗ ████║██╔═══██╗██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
#  ██╔████╔██║██║   ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
#  ██║╚██╔╝██║██║   ██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
#  ██║ ╚═╝ ██║╚██████╔╝███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
#  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
#
# moewatch/hooks/manager.py
#
# HookManager — owns the full lifecycle of all PyTorch forward hooks attached
# to router modules. Guarantees clean teardown in every code path.
#
# Responsibilities
# ----------------
#   - Resolve module references from string names via model.named_modules().
#   - Register one RouterHook per router module.
#   - Implement context-manager protocol so hooks are always removed on exit,
#     even when an exception propagates through a training step.
#   - Expose attach() / detach() for imperative usage (MoEWatch path).
#   - Track registered handles for guaranteed cleanup without double-removal.
#
# Invariants
# ----------
#   - After detach() or __exit__, zero hooks remain registered on the model.
#   - attach() is idempotent: calling it twice on the same instance is a no-op
#     with a warning rather than registering duplicate hooks.
#   - The model is never mutated — no parameter changes, no attribute injection.
#
# Author : Abinesh (GitHub: Abineshabee)
# License: Apache 2.0
# Version: 0.1.0
# ------------------------------------------------------------------------------

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import torch.nn as nn

from ..config import WatchConfig
from .router_hook import RouterHook

log = logging.getLogger(__name__)


class HookManager:
    """Manages the full lifecycle of moewatch's PyTorch forward hooks.

    Parameters
    ----------
    model              : The MoE model to instrument (never modified).
    router_module_names: Fully-qualified module names to hook
                         (e.g. ``["model.layers.0.block_sparse_moe"]``).
    collector          : ``StatCollector`` instance that receives ``RoutingEvent`` objects.
    config             : ``WatchConfig`` — forwarded to each ``RouterHook``.

    Examples
    --------
    Context-manager usage (recommended — guaranteed cleanup):

    >>> with HookManager(model, router_names, collector, config) as hm:
    ...     for batch in dataloader:
    ...         model(**batch)   # hooks fire here

    Imperative usage (MoEWatch training-time path):

    >>> hm = HookManager(model, router_names, collector, config)
    >>> hm.attach()
    >>> # ... training ...
    >>> hm.detach()   # always call this, even on exception
    """

    def __init__(
        self,
        model: nn.Module,
        router_module_names: List[str],
        collector: Any,
        config: WatchConfig,
    ) -> None:
        self.model               = model
        self.router_module_names = list(router_module_names)
        self.collector           = collector
        self.config              = config

        # name → RouterHook
        self._hooks:   Dict[str, RouterHook] = {}
        # name → torch hook handle (returned by register_forward_hook)
        self._handles: Dict[str, Any]        = {}
        self._attached: bool                 = False

    # --------------------------------------------------------------------------
    # Attach
    # --------------------------------------------------------------------------

    def attach(self) -> "HookManager":
        """Register all router hooks on the model.

        Returns self for method chaining. Idempotent — calling attach() when
        already attached emits a warning and returns immediately.

        Raises
        ------
        ValueError
            If a module name in ``router_module_names`` cannot be found in
            the model's named modules. Partial attachment does not occur —
            if any name is invalid the entire attach is aborted and any
            already-registered handles for this call are cleaned up.
        """
        if self._attached:
            warnings.warn(
                "[moewatch] HookManager.attach() called on an already-attached "
                "instance. Ignoring. Call detach() first to re-attach.",
                UserWarning,
                stacklevel=2,
            )
            return self

        # -- Build name → module map -----------------------------------------
        module_map: Dict[str, nn.Module] = {
            name: module
            for name, module in self.model.named_modules()
        }

        # -- Validate all names before touching the model --------------------
        missing = [
            name for name in self.router_module_names
            if name not in module_map
        ]
        if missing:
            missing_str = "\n  ".join(missing)
            available_sample = "\n  ".join(
                list(module_map.keys())[:10]
            )
            raise ValueError(
                f"[moewatch] The following router module name(s) were not found "
                f"in the model:\n  {missing_str}\n\n"
                f"Available module names (first 10):\n  {available_sample}\n\n"
                f"Tip: list all modules with:\n"
                f"  for name, _ in model.named_modules(): print(name)"
            )

        # -- Register hooks --------------------------------------------------
        newly_registered: Dict[str, Any] = {}
        try:
            for name in self.router_module_names:
                module = module_map[name]
                hook   = RouterHook(
                    layer_name=name,
                    collector=self.collector,
                    config=self.config,
                )
                handle = module.register_forward_hook(hook)
                self._hooks[name]   = hook
                newly_registered[name] = handle
                log.debug("[moewatch] Hook attached: %s", name)

        except Exception:
            # Roll back any hooks registered in this call before re-raising
            for h in newly_registered.values():
                try:
                    h.remove()
                except Exception:   # pragma: no cover
                    pass
            self._hooks.clear()
            raise

        self._handles  = newly_registered
        self._attached = True

        log.info(
            "[moewatch] HookManager attached %d hook(s).",
            len(self._handles),
        )
        return self

    # --------------------------------------------------------------------------
    # Detach
    # --------------------------------------------------------------------------

    def detach(self) -> "HookManager":
        """Remove all registered hooks from the model.

        Safe to call multiple times — subsequent calls are no-ops. Guaranteed
        not to raise: individual handle removal errors are logged and suppressed
        so that a partial failure cannot leave the caller in an unrecoverable
        state.

        Returns self for method chaining.
        """
        if not self._attached:
            return self

        failed: List[str] = []
        for name, handle in self._handles.items():
            try:
                handle.remove()
                log.debug("[moewatch] Hook detached: %s", name)
            except Exception as exc:                    # pragma: no cover
                failed.append(f"{name}: {exc}")

        if failed:                                      # pragma: no cover
            log.warning(
                "[moewatch] Non-fatal errors during hook removal:\n  %s",
                "\n  ".join(failed),
            )

        self._handles.clear()
        self._hooks.clear()
        self._attached = False

        log.info("[moewatch] HookManager detached all hooks.")
        return self

    # --------------------------------------------------------------------------
    # Context-manager protocol
    # --------------------------------------------------------------------------

    def __enter__(self) -> "HookManager":
        self.attach()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Always detach on exit — even if an exception propagated."""
        self.detach()
        # Return None (falsy) — do not suppress exceptions

    # --------------------------------------------------------------------------
    # Introspection
    # --------------------------------------------------------------------------

    @property
    def is_attached(self) -> bool:
        """True if hooks are currently active on the model."""
        return self._attached

    @property
    def hook_count(self) -> int:
        """Number of hooks currently registered."""
        return len(self._handles)

    def get_hook(self, layer_name: str) -> Optional[RouterHook]:
        """Return the ``RouterHook`` for a given layer name, or ``None``."""
        return self._hooks.get(layer_name)

    def diagnostics(self) -> Dict[str, Dict[str, int]]:
        """Return per-hook call and error counts.

        Returns
        -------
        dict
            ``{layer_name: {"calls": int, "errors": int}}``
        """
        return {
            name: {"calls": hook.call_count, "errors": hook.error_count}
            for name, hook in self._hooks.items()
        }

    def __repr__(self) -> str:
        status = "attached" if self._attached else "detached"
        return (
            f"HookManager("
            f"hooks={len(self._handles)}, "
            f"status={status}"
            f")"
        )
