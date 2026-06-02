# ------------------------------------------------------------------------------
# moewatch.hooks
# Hook lifecycle management and router module auto-detection.
# ------------------------------------------------------------------------------

from .manager     import HookManager
from .router_hook import RouterHook, RoutingEvent
from .detection   import detect_router_modules

__all__ = [
    "HookManager",
    "RouterHook",
    "RoutingEvent",
    "detect_router_modules",
]
