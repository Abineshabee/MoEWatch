# moewatch API Reference

> **Version:** 0.1.0 · [GitHub](https://github.com/Abineshabee/moewatch) · Apache 2.0

This document covers every public class, function, and data type in moewatch. Internal helpers (prefixed with `_`) are excluded unless they're useful for advanced usage.

---

## Contents

1. [Top-level functions](#1-top-level-functions)
   - [`audit()`](#audit)
2. [MoEWatch](#2-moewatch)
   - [`MoEWatch`](#moewatch-class)
   - [`MoEWatchCallback`](#moeewatchcallback)
   - [`Alert`](#alert)
3. [Configuration](#3-configuration)
   - [`WatchConfig`](#watchconfig)
   - [`OutputMode`](#outputmode)
   - [`AlertLevel`](#alertlevel)
4. [AuditReport](#4-auditreport)
   - [`AuditReport`](#auditreport-class)
   - [`OverallHealth`](#overallhealth)
   - [`DeadExpertEntry`](#deadexpertentry)
   - [`UtilizationSummary`](#utilizationsummary)
5. [Analyzer layer](#5-analyzer-layer)
   - [`EntropyAnalyzer`](#entropyanalyzer)
   - [`EntropyResult`](#entropyresult)
   - [`LayerEntropyReport`](#layerentropyreport)
   - [`TrendDirection`](#trenddirection)
   - [`CollapseDetector`](#collapsedetector)
   - [`ExpertStatus`](#expertstatus)
   - [`ExpertState`](#expertstate)
   - [`LayerCollapseReport`](#layercollapsereport)
6. [Collector layer](#6-collector-layer)
   - [`StatCollector`](#statcollector)
   - [`LayerStats`](#layerstats)
   - [`RingBuffer`](#ringbuffer)
7. [Hooks layer](#7-hooks-layer)
   - [`HookManager`](#hookmanager)
   - [`RouterHook`](#routerhook)
   - [`RoutingEvent`](#routingevent)
   - [`detect_router_modules()`](#detect_router_modules)
8. [Entropy utilities](#8-entropy-utilities)
   - [`compute_entropy()`](#compute_entropy)
   - [`compute_entropy_from_logits()`](#compute_entropy_from_logits)
   - [`max_entropy()`](#max_entropy)
   - [`normalised_entropy()`](#normalised_entropy)
9. [Report rendering](#9-report-rendering)
   - [`CLIReporter`](#clireporter)

---

## 1. Top-level functions

### `audit()`

```python
moewatch.audit(
    model,
    dataloader=None,
    *,
    steps=50,
    config=None,
    use_no_grad=True,
    verbose=True,
) -> AuditReport
```

Run a full offline diagnostic pass on a HuggingFace MoE model. This is the primary entry point for one-shot diagnostics — no training loop modification required.

**What it does:**

1. Auto-detects (or accepts manual) router modules via `HookManager`.
2. Runs `steps` forward passes through the model under `torch.no_grad()`.
3. Collects per-layer routing statistics into a `StatCollector`.
4. Runs `EntropyAnalyzer` and `CollapseDetector` over the collected stats.
5. Returns a structured `AuditReport`. Hooks are always cleaned up, even on exception.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module` | required | Any HuggingFace MoE model. Never modified. |
| `dataloader` | iterable, optional | `None` | Standard `DataLoader` or any iterable yielding `dict`, `tuple`, or `Tensor` batches. When `None`, moewatch synthesises a random batch from the model's embedding vocabulary. |
| `steps` | `int` | `50` | Number of forward passes to run. Use `10` for quick checks, `200` for publication-quality audits. |
| `config` | `WatchConfig`, optional | `None` | Full configuration. Defaults to `WatchConfig()` when not provided. |
| `use_no_grad` | `bool` | `True` | Wraps forward passes in `torch.no_grad()`. Set `False` only when auditing inside a step that requires gradients. |
| `verbose` | `bool` | `True` | Print a progress header to stdout. Set `False` for programmatic use. |

**Returns:** [`AuditReport`](#auditreport-class)

**Raises**

- `TypeError` — `model` is not a `torch.nn.Module`.
- `ValueError` — `steps < 1`, or the dataloader is empty.
- `RuntimeError` — no router modules detected and no manual override provided.

**Warns**

- `UserWarning` — if no dataloader is provided (synthetic data used).
- `UserWarning` — if the dataloader is exhausted before `steps` passes complete.
- `UserWarning` — if fewer than 10 forward passes completed (results unreliable).

**Examples**

```python
# Minimal — synthetic data, defaults
report = moewatch.audit(model)

# With a real DataLoader
report = moewatch.audit(model, train_loader, steps=100)

# Silent mode for CI — no output, inspect the report object
from moewatch import audit, WatchConfig
report = audit(model, config=WatchConfig.silent())
assert not report.has_collapse, f"Expert collapse detected: {report.dead_experts()}"

# Manual router override for custom architectures
config = WatchConfig(router_modules=["model.moe.gate", "model.moe2.gate"])
report = audit(model, loader, config=config)
```

---

## 2. MoEWatch

### `MoEWatch` class

```python
moewatch.MoEWatch(model, config=None)
```

Live training-time MoE diagnostic monitor. Attaches PyTorch forward hooks to router modules and emits structured `Alert` objects whenever routing health degrades.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module` | required | The MoE model to monitor. Never modified. |
| `config` | `WatchConfig`, optional | `None` | Diagnostic configuration. Defaults to `WatchConfig()`. |

**Raises:** `TypeError` if `model` is not a `torch.nn.Module`.

---

#### Methods

##### `MoEWatch.start() -> MoEWatch`

Attach hooks and begin monitoring. Returns `self` for method chaining.

Resolves router modules (auto-detected or from `config.router_modules`), wires up the `StatCollector` and `HookManager`, and prints a startup banner. Idempotent — calling `start()` on an already-running watcher emits a `UserWarning` and returns immediately.

**Raises:** `RuntimeError` if no router modules can be detected.

---

##### `MoEWatch.stop() -> MoEWatch`

Detach all hooks and stop monitoring. Returns `self` for method chaining.

Safe to call multiple times (subsequent calls are no-ops). Guaranteed not to raise — any individual handle removal error is logged and suppressed so no hooks leak.

---

##### `MoEWatch.attach(trainer) -> MoEWatch`

Attach moewatch to a HuggingFace `Trainer`. Calls `start()` then injects a `MoEWatchCallback` into the trainer's callback list.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `trainer` | `transformers.Trainer` | Any HuggingFace Trainer. Must have an `add_callback` method. |

**Returns:** `self`  
**Raises:** `TypeError` if `trainer` has no `add_callback` method.

---

##### `MoEWatch.detach() -> MoEWatch`

Alias for `stop()`. Consistent with PyTorch hook terminology.

---

##### `MoEWatch.step(global_step) -> List[Alert]`

Run a diagnostic check at the given training step. Called automatically by `MoEWatchCallback` when using the HuggingFace Trainer integration; call manually from a custom loop.

Checks fire only every `config.log_every` steps (other steps return `[]` immediately). When nothing is wrong, a single `INFO` heartbeat alert is returned.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `global_step` | `int` | Current training step. Used in alert messages and the alert log. |

**Returns:** `List[Alert]` — alerts fired at this step (empty when watcher is not attached or step is skipped).

---

##### `MoEWatch.get_alert_log() -> List[Alert]`

Return a copy of all alerts emitted since monitoring started. Mutating the returned list has no effect on the internal log.

---

##### `MoEWatch.get_alert_log_json() -> str`

Return all alerts as a JSON array string. Each alert is serialised via `Alert.to_dict()`.

---

##### `MoEWatch.summary() -> str`

Return a one-line text summary: total alerts, error count, warn count, info count.

---

#### Properties

| Property | Type | Description |
|---|---|---|
| `is_attached` | `bool` | `True` when hooks are currently active. |
| `global_step` | `int` | Last training step seen by the watcher. |

---

#### Context manager

`MoEWatch` implements `__enter__` / `__exit__`, calling `start()` and `stop()` respectively. This guarantees hook cleanup even on exception.

```python
with MoEWatch(model, config=config) as watcher:
    for step, batch in enumerate(dataloader):
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        watcher.step(step)
```

---

#### Examples

**HuggingFace Trainer:**

```python
from moewatch import MoEWatch, WatchConfig

watcher = MoEWatch(model, config=WatchConfig(log_every=100))
watcher.attach(trainer)
trainer.train()
watcher.detach()

# Inspect alert history after training
for alert in watcher.get_alert_log():
    if alert.level.value == "ERROR":
        print(f"Step {alert.step}: {alert.message}")
```

**Custom training loop:**

```python
watcher = MoEWatch(model)
watcher.start()

for step, batch in enumerate(dataloader):
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    alerts = watcher.step(step)
    for a in alerts:
        if a.level.value == "ERROR":
            print(f"⚠ {a.message}")

watcher.stop()
```

---

### `MoEWatchCallback`

```python
moewatch.MoEWatchCallback(watcher)
```

HuggingFace `TrainerCallback` that ticks `MoEWatch.step()` at each logging event. You do not normally instantiate this directly — `MoEWatch.attach(trainer)` creates and registers it for you.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `watcher` | `MoEWatch` | The parent watcher. Must already be started before the callback fires. |

**Callback methods (called by HuggingFace Trainer internally):**

- `on_log(args, state, control, **kwargs)` — calls `watcher.step(state.global_step)` at each logging step.
- `on_train_end(args, state, control, **kwargs)` — calls `watcher.stop()` when training completes.

---

### `Alert`

```python
@dataclass
moewatch.Alert
```

A single diagnostic event emitted by `MoEWatch.step()`.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `step` | `int` | Training step at which the alert fired. |
| `level` | `AlertLevel` | Severity: `INFO`, `WARN`, or `ERROR`. |
| `layer_name` | `str` | Fully-qualified name of the affected router module. `"*"` for model-wide heartbeat alerts. |
| `message` | `str` | Human-readable description of the condition. |
| `metric` | `float`, optional | The numeric value that triggered the alert (e.g. utilisation fraction, entropy norm, imbalance ratio). |
| `suggestion` | `str`, optional | Actionable fix suggestion. Only present on `ERROR`-level alerts. |
| `timestamp` | `float` | Wall-clock time (seconds since epoch) when the alert was created. |

**Methods**

- `to_dict() -> dict` — JSON-serialisable dictionary representation.
- `to_json_line() -> str` — Single-line JSON string (for newline-delimited JSON streams).

---

## 3. Configuration

### `WatchConfig`

```python
@dataclass
moewatch.WatchConfig(
    dead_threshold=0.001,
    cold_threshold=0.005,
    cold_steps_limit=500,
    entropy_warn=0.60,
    entropy_critical=0.40,
    entropy_drop_warn=0.20,
    load_imbalance_warn=3.0,
    load_imbalance_error=5.0,
    window_steps=500,
    sample_every=10,
    log_every=100,
    ring_buffer_capacity=10_000,
    output=OutputMode.CONSOLE,
    no_color=False,
    router_modules=[],
)
```

Central configuration object for all moewatch components. All fields carry empirically-derived defaults from the Switch Transformer, Mixtral, and DeepSeek-MoE training literature.

**Expert health thresholds**

| Field | Type | Default | Description |
|---|---|---|---|
| `dead_threshold` | `float` | `0.001` | Token utilisation fraction at or below which an expert is classified DEAD (0.1%). Based on Switch Transformer findings where experts below this never recovered. |
| `cold_threshold` | `float` | `0.005` | Token utilisation fraction at or below which an expert is classified COLD (0.5%). Must be > `dead_threshold`. |
| `cold_steps_limit` | `int` | `500` | Number of consecutive analysis steps an expert may remain COLD before being promoted to DEAD. |

**Entropy thresholds**

| Field | Type | Default | Description |
|---|---|---|---|
| `entropy_warn` | `float` | `0.60` | Fraction of H_max below which a WARN is emitted (60% of log₂(n_experts)). |
| `entropy_critical` | `float` | `0.40` | Fraction of H_max below which an ERROR is emitted (40%). Must be < `entropy_warn`. |
| `entropy_drop_warn` | `float` | `0.20` | Relative entropy drop over `window_steps` that triggers a WARN (20% relative drop). |

**Load imbalance thresholds**

| Field | Type | Default | Description |
|---|---|---|---|
| `load_imbalance_warn` | `float` | `3.0` | Ratio of max expert utilisation to mean utilisation above which a WARN fires. |
| `load_imbalance_error` | `float` | `5.0` | Same ratio, above which an ERROR fires. Must be > `load_imbalance_warn`. |

**Sampling and windowing**

| Field | Type | Default | Description |
|---|---|---|---|
| `window_steps` | `int` | `500` | Rolling window size for trend analysis and dead-expert duration tracking. |
| `sample_every` | `int` | `10` | Instrument every Nth forward pass. Set to `1` for maximum fidelity (~2–4% overhead). |
| `log_every` | `int` | `100` | Emit a live alert summary every N training steps. |
| `ring_buffer_capacity` | `int` | `10_000` | Maximum `RoutingEvent` entries held in memory. Older events are overwritten when full. |

**Output and display**

| Field | Type | Default | Description |
|---|---|---|---|
| `output` | `OutputMode` or `str` | `OutputMode.CONSOLE` | Controls real-time diagnostic output. Accepts enum values or string equivalents: `"console"`, `"json"`, `"silent"`. |
| `no_color` | `bool` | `False` | Disable ANSI colour codes. Also auto-enabled when the `NO_COLOR` environment variable is set. |

**Router discovery**

| Field | Type | Default | Description |
|---|---|---|---|
| `router_modules` | `List[str]` | `[]` | Explicit fully-qualified module names to instrument. When non-empty, auto-detection is bypassed entirely. |

**Validation:** `WatchConfig` validates all fields in `__post_init__` and raises `ValueError` with a descriptive message for any logically inconsistent configuration (e.g. `dead_threshold >= cold_threshold`). A `UserWarning` is emitted (but not raised) for `sample_every=1` or when `ring_buffer_capacity < window_steps`.

---

#### Class methods (presets)

```python
WatchConfig.default()      # equivalent to WatchConfig() — recommended starting point
WatchConfig.aggressive()   # tighter thresholds, sample_every=1 — use for debugging collapse
WatchConfig.lightweight()  # sample_every=50, ring_buffer_capacity=2_000 — production runs
WatchConfig.silent()       # output=OutputMode.SILENT — no real-time output
```

**`aggressive()` values:** `dead_threshold=0.0005`, `cold_threshold=0.002`, `entropy_warn=0.70`, `entropy_critical=0.50`, `entropy_drop_warn=0.10`, `sample_every=1`, `log_every=50`, `window_steps=200`.

**`lightweight()` values:** `sample_every=50`, `log_every=500`, `ring_buffer_capacity=2_000`.

---

#### Instance methods

##### `WatchConfig.to_dict() -> dict`

Return a JSON-serialisable dictionary of all configuration fields. Useful for logging config alongside `AuditReport.to_json()`.

---

### `OutputMode`

```python
class moewatch.OutputMode(str, Enum)
```

Controls how moewatch emits diagnostic output.

| Value | Description |
|---|---|
| `OutputMode.CONSOLE` | Human-readable coloured ASCII to stdout. Default. |
| `OutputMode.JSON` | Newline-delimited JSON — suitable for Grafana, Splunk, or custom log pipelines. |
| `OutputMode.SILENT` | Suppresses all real-time output. Results only available via `AuditReport`. |

---

### `AlertLevel`

```python
class moewatch.AlertLevel(str, Enum)
```

Severity levels for diagnostic alerts. Follows the standard INFO → WARN → ERROR escalation ladder. moewatch never emits FATAL — it diagnoses, it does not stop your training run.

| Value | Description |
|---|---|
| `AlertLevel.INFO` | Routing health nominal. |
| `AlertLevel.WARN` | Degraded routing — investigate soon. |
| `AlertLevel.ERROR` | Severe collapse or imbalance — intervention recommended. |

---

## 4. AuditReport

### `AuditReport` class

```python
moewatch.AuditReport
```

Structured, immutable diagnostic result returned by `audit()`. Holds the complete output of a single audit run. Safe to cache, pass between threads, or serialise to JSON.

Not normally constructed directly — use `audit()`.

---

#### Primary query methods

##### `AuditReport.dead_experts(*, include_cold=True) -> List[DeadExpertEntry]`

Return a flat list of all dead (and optionally cold) experts, ordered by `(layer_idx, expert_idx)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `include_cold` | `bool` | `True` | Include COLD experts alongside DEAD ones. |

Returns an empty list when all experts are healthy. Never returns `None`.

```python
dead = report.dead_experts()
for entry in dead:
    print(f"Layer {entry.layer_idx}  Expert {entry.expert_idx}  "
          f"{entry.state.value}  util={entry.utilization_pct:.3f}%")

# Only confirmed-dead, not cold:
confirmed = report.dead_experts(include_cold=False)
```

---

##### `AuditReport.routing_entropy() -> Dict[str, EntropyResult]`

Return per-layer entropy results as `{layer_name: EntropyResult}`. Empty dict when no events were collected. Never returns `None`.

```python
for name, result in report.routing_entropy().items():
    print(f"{name}: {result.entropy_norm:.1%} of max  [{result.alert_level.value}]  {result.trend}")
```

---

##### `AuditReport.utilization() -> Dict[str, UtilizationSummary]`

Return per-expert token distribution for every layer as `{layer_name: UtilizationSummary}`. Empty dict when no events were collected. Never returns `None`.

```python
for name, u in report.utilization().items():
    print(f"{name}: imbalance={u.load_imbalance_score:.2f}x  "
          f"max={u.max_util*100:.1f}%  min={u.min_util*100:.3f}%")
```

---

##### `AuditReport.recommendations() -> List[str]`

Return a prioritised list of actionable fix suggestions derived from the diagnostic state, ordered from most to least severe.

Returns a single `[INFO]` string when the model is healthy. Each string is self-contained and actionable.

```python
for rec in report.recommendations():
    print(f"  💡 {rec}")
```

---

##### `AuditReport.summary() -> str`

Return a concise multi-line plain-text summary string (no ANSI codes). Includes overall health, per-layer entropy, expert collapse table, and recommendations. Useful for logging or notebook display.

---

#### Per-layer accessors

##### `AuditReport.layer_names() -> List[str]`

Return the ordered list of instrumented router module names.

##### `AuditReport.layer_stats(layer_name) -> LayerStats | None`

Return raw `LayerStats` for a single layer, or `None` if not found.

##### `AuditReport.entropy_for(layer_name) -> EntropyResult | None`

Return the `EntropyResult` for a single layer, or `None`.

##### `AuditReport.collapse_for(layer_name) -> LayerCollapseReport | None`

Return the `LayerCollapseReport` for a single layer, or `None`.

---

#### Aggregate properties

| Property | Type | Description |
|---|---|---|
| `overall_health` | `OverallHealth` | Top-level routing health classification. |
| `n_layers` | `int` | Total number of instrumented router layers. |
| `n_dead` | `int` | Total confirmed dead experts across all layers. |
| `n_cold` | `int` | Total cold (early-warning) experts across all layers. |
| `n_healthy` | `int` | Total healthy experts across all layers. |
| `n_experts_total` | `int` | Total expert count across all layers. |
| `worst_entropy_layer` | `EntropyResult \| None` | Layer with the lowest normalised entropy. |
| `has_collapse` | `bool` | `True` when at least one expert is confirmed DEAD. |
| `has_warnings` | `bool` | `True` when at least one expert is COLD or one layer's entropy is in WARN range. |
| `completed_steps` | `int` | Number of forward passes that ran (may be < requested). |
| `elapsed_seconds` | `float` | Wall-clock time the audit loop took. |
| `device` | `str` | Device used (e.g. `"cuda:0"`, `"cpu"`). |
| `router_module_names` | `List[str]` | Ordered list of all instrumented module names. |
| `config` | `WatchConfig` | Configuration snapshot used for this audit. |
| `created_at` | `float` | Wall-clock creation timestamp. |

---

#### Serialisation

##### `AuditReport.to_json(indent=2) -> str`

Serialise the full report to a UTF-8 JSON string. All tensor data is converted to Python lists; `NaN` values become `null`.

```python
with open("moewatch_report.json", "w") as f:
    f.write(report.to_json())
```

##### `AuditReport.to_json_file(path, indent=2) -> None`

Write the full report to a JSON file. The parent directory must exist.

```python
report.to_json_file("./reports/audit_2026_06.json")
```

**JSON structure:**

```json
{
  "metadata":        { "overall_health": "HEALTHY", "n_layers": 32, ... },
  "config":          { "dead_threshold": 0.001, ... },
  "entropy":         { "global_alert": "INFO", "layers": { "model.layers.0...": {...} } },
  "collapse":        { "model.layers.0...": { "n_dead": 0, "experts": [...] } },
  "dead_experts":    [],
  "utilization":     { "model.layers.0...": { "utilization": [...], ... } },
  "recommendations": ["[INFO] All experts active..."]
}
```

---

### `OverallHealth`

```python
class moewatch.OverallHealth(str, Enum)
```

Top-level routing health classification for the entire model, derived from the union of all per-layer entropy and collapse results.

| Value | Description |
|---|---|
| `HEALTHY` | All experts active, entropy above `entropy_warn` on every layer. No action required. |
| `DEGRADING` | At least one expert is COLD or one layer's entropy is in WARN range. Monitor closely. |
| `CRITICAL` | At least one expert is DEAD or one layer's entropy is in ERROR range. Immediate intervention recommended. |
| `UNKNOWN` | No events were collected. Check model and dataloader configuration. |

---

### `DeadExpertEntry`

```python
@dataclass(frozen=True)
moewatch.DeadExpertEntry
```

Flat, user-facing record for a single dead or cold expert. Produced by `AuditReport.dead_experts()`.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `layer_name` | `str` | Fully-qualified module name of the router layer. |
| `expert_idx` | `int` | Zero-based expert index within the layer. |
| `state` | `ExpertState` | `DEAD` or `COLD`. |
| `utilization` | `float` | Fraction of total tokens routed to this expert (0.0–1.0). |
| `utilization_pct` | `float` | `utilization * 100` — convenience field for display. |
| `consecutive_cold_steps` | `int` | Number of analysis calls during which the expert has been non-healthy. |
| `layer_idx` | `int` | Positional index of this layer in the ordered router list (0-based). |

**Properties:** `is_dead`, `is_cold`.  
**Methods:** `to_dict() -> dict`.

---

### `UtilizationSummary`

```python
@dataclass(frozen=True)
moewatch.UtilizationSummary
```

Expert token-distribution summary for a single router layer. Produced by `AuditReport.utilization()`.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `layer_name` | `str` | Fully-qualified module name. |
| `layer_idx` | `int` | Positional index in the ordered router list. |
| `n_experts` | `int` | Total number of experts. |
| `utilization` | `List[float]` | Per-expert token fraction (0.0–1.0). Length == `n_experts`. |
| `utilization_pct` | `List[float]` | Per-expert token percentage (0.0–100.0). |
| `token_counts` | `List[int]` | Raw per-expert token counts over the audit window. |
| `total_tokens` | `int` | Sum of all token counts. |
| `max_util` | `float` | Utilisation of the most-loaded expert. |
| `min_util` | `float` | Utilisation of the least-loaded expert. |
| `mean_util` | `float` | Mean expert utilisation (== 1 / n_experts for perfect balance). |
| `load_imbalance_score` | `float` | max_util / mean_util. 1.0 = perfect balance. `nan` when no data. |
| `event_count` | `int` | Number of `RoutingEvent` objects that contributed to these stats. |
| `step_range` | `Tuple[int, int]` | `(first_step, last_step)` of contributing events. |

**Methods:** `to_dict() -> dict`.

---

## 5. Analyzer layer

### `EntropyAnalyzer`

```python
moewatch.analyzer.EntropyAnalyzer(config)
```

Stateful per-layer Shannon entropy analyser. Consumes `Dict[str, LayerStats]` from `StatCollector` and emits a `LayerEntropyReport`.

Maintains a per-layer entropy history deque (up to 50 values) for trend detection. A single instance can be shared across both `audit()` and `MoEWatch` paths.

**Parameters:** `config: WatchConfig`

---

#### Methods

##### `EntropyAnalyzer.analyze(all_stats) -> LayerEntropyReport`

Run entropy analysis over all tracked layers.

**Parameters:** `all_stats: Dict[str, LayerStats]` — as returned by `StatCollector.get_all_stats()`. An empty dict is valid.

**Returns:** `LayerEntropyReport`

**Computation source priority:**
1. Raw logits (if `LayerStats.has_raw_logits` is `True`) — more precise, captures the full router distribution.
2. Token counts — always available; accurate for large batches.

---

##### `EntropyAnalyzer.reset_history(layer_name=None) -> None`

Clear entropy history for a specific layer or all layers. Call when a model checkpoint is restored mid-run.

---

##### `EntropyAnalyzer.history_for(layer_name) -> List[float]`

Return the normalised entropy history list for a layer (oldest first). Returns an empty list if the layer has never been analysed.

---

#### Properties

| Property | Type | Description |
|---|---|---|
| `tracked_layers` | `List[str]` | Names of all layers that have at least one history point. |

---

### `EntropyResult`

```python
@dataclass(frozen=True)
moewatch.analyzer.EntropyResult
```

Immutable entropy snapshot for a single MoE router layer. Produced by `EntropyAnalyzer.analyze()`.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `layer_name` | `str` | Fully-qualified module name. |
| `n_experts` | `int` | Number of experts in this layer. |
| `entropy_bits` | `float` | Absolute Shannon entropy in bits. Range: [0, log₂(n_experts)]. |
| `entropy_norm` | `float` | Entropy normalised to [0, 1] relative to H_max. 1.0 = perfectly uniform; 0.0 = fully collapsed. |
| `h_max` | `float` | Theoretical maximum entropy (log₂(n_experts)). |
| `alert_level` | `AlertLevel` | Severity of the entropy alert. |
| `trend` | `str` | One of `TrendDirection` constants — direction of entropy change. |
| `trend_delta` | `float` | Relative entropy change over the history window: `(H_now - H_prev) / H_prev`. Positive = improving; negative = declining. |
| `source` | `str` | `"logits"` when computed from raw router logits; `"counts"` when estimated from token-count distribution. |
| `event_count` | `int` | Number of `RoutingEvent` objects that contributed. |
| `is_empty` | `bool` | `True` when no events were available. |

**Properties:** `is_healthy`, `is_critical`, `is_declining`.  
**Methods:** `to_dict() -> dict`.

---

### `LayerEntropyReport`

```python
@dataclass
moewatch.analyzer.LayerEntropyReport
```

Entropy report for all tracked MoE layers. Returned by `EntropyAnalyzer.analyze()`.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `results` | `Dict[str, EntropyResult]` | `{layer_name: EntropyResult}` — one per layer. |
| `global_alert` | `AlertLevel` | Worst alert level across all layers. |
| `n_warn` | `int` | Number of layers in WARN state. |
| `n_error` | `int` | Number of layers in ERROR state. |
| `n_declining` | `int` | Number of layers with DECLINING entropy trend. |

**Properties:** `layers`, `is_healthy`.

**Methods:**

- `worst_layer() -> EntropyResult | None` — Layer with the lowest normalised entropy.
- `layers_below(threshold_norm) -> List[EntropyResult]` — All non-empty layers whose normalised entropy is below `threshold_norm`.
- `declining_layers() -> List[EntropyResult]` — All layers with a DECLINING trend.
- `to_dict() -> dict`.

---

### `TrendDirection`

```python
moewatch.analyzer.TrendDirection
```

String constants (not an Enum) for entropy trend direction.

| Constant | Value | Description |
|---|---|---|
| `TrendDirection.STABLE` | `"STABLE"` | Entropy is not changing significantly. |
| `TrendDirection.IMPROVING` | `"IMPROVING"` | Entropy is increasing (healthy recovery). |
| `TrendDirection.DECLINING` | `"DECLINING"` | Entropy is decreasing (collapse precursor). Triggers WARN. |
| `TrendDirection.UNKNOWN` | `"UNKNOWN"` | Fewer than 4 history points — no trend verdict yet. |

---

### `CollapseDetector`

```python
moewatch.analyzer.CollapseDetector(config)
```

Stateful expert collapse detector. Classifies each expert as HEALTHY, COLD, or DEAD based on utilisation thresholds and persistence tracking across calls.

Maintains per-layer, per-expert consecutive cold step counters (`_cold_counters`). On each `detect()` call, counters are incremented for cold experts and reset to 0 for recovered experts. An expert cold for more than `config.cold_steps_limit` steps is promoted to DEAD.

**Parameters:** `config: WatchConfig`

---

#### Methods

##### `CollapseDetector.detect(all_stats) -> Dict[str, LayerCollapseReport]`

Run collapse detection over all tracked layers.

**Parameters:** `all_stats: Dict[str, LayerStats]` — as returned by `StatCollector.get_all_stats()`.

**Returns:** `Dict[str, LayerCollapseReport]` — same key set as `all_stats`.

**Classification rules per expert:**
1. `utilisation ≤ dead_threshold` → DEAD immediately.
2. `utilisation ≤ cold_threshold` → cold counter incremented; DEAD if counter ≥ `cold_steps_limit`; COLD otherwise.
3. `utilisation > cold_threshold` → HEALTHY, counter reset to 0.

---

##### `CollapseDetector.reset(layer_name=None) -> None`

Reset cold-step counters for a specific layer or all layers. Call when restoring a checkpoint mid-training.

---

##### `CollapseDetector.cold_counter_for(layer_name, expert_idx) -> int`

Return the consecutive cold-step count for a specific expert. Returns 0 if not tracked.

---

##### `CollapseDetector.compute_load_imbalance(utilization) -> Tuple[float, int, float]`  *(static)*

Compute the load imbalance score for a utilisation tensor.

**Parameters:** `utilization: torch.Tensor` — 1-D float tensor of shape `(n_experts,)`.

**Returns:** `(load_imbalance_score, argmax_expert_idx, max_utilization)`  
Returns `(nan, -1, nan)` for empty or all-zero tensors.

---

#### Properties

| Property | Type | Description |
|---|---|---|
| `tracked_layers` | `List[str]` | Names of layers with at least one cold-step counter entry. |

---

### `ExpertStatus`

```python
@dataclass(frozen=True)
moewatch.analyzer.ExpertStatus
```

Immutable diagnostic snapshot for a single expert within a router layer. Stored in `LayerCollapseReport.experts`.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `layer_name` | `str` | Fully-qualified module name of the parent router layer. |
| `expert_idx` | `int` | Zero-based expert index. |
| `state` | `ExpertState` | Current health classification. |
| `alert_level` | `AlertLevel` | DEAD → ERROR, COLD → WARN, HEALTHY → INFO. |
| `utilization` | `float` | Fraction of total tokens routed to this expert. `nan` when no events. |
| `token_count` | `int` | Raw token count for this expert in the current window. |
| `consecutive_cold_steps` | `int` | Number of consecutive analysis calls during which the expert was non-healthy. |

**Properties:** `is_healthy`, `is_cold`, `is_dead`, `is_problematic`.  
**Methods:** `to_dict() -> dict`.

---

### `ExpertState`

```python
class moewatch.analyzer.ExpertState(str, Enum)
```

Health state for a single MoE expert.

| Value | Description |
|---|---|
| `HEALTHY` | Actively receiving tokens above `cold_threshold`. |
| `COLD` | Utilisation dropped below `cold_threshold` but above `dead_threshold`. May recover; triggers WARN. |
| `DEAD` | At or below `dead_threshold`, or COLD for more than `cold_steps_limit` steps. Triggers ERROR. |
| `UNKNOWN` | No routing data collected yet. |

---

### `LayerCollapseReport`

```python
@dataclass
moewatch.analyzer.LayerCollapseReport
```

Expert collapse diagnostics for a single MoE router layer. Produced by `CollapseDetector.detect()`.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `layer_name` | `str` | Fully-qualified module name. |
| `n_experts` | `int` | Total number of experts. |
| `experts` | `List[ExpertStatus]` | Per-expert status, ordered by expert index. |
| `n_dead` | `int` | Confirmed dead experts. |
| `n_cold` | `int` | Cold (early-warning) experts. |
| `n_healthy` | `int` | Healthy experts. |
| `global_alert` | `AlertLevel` | Worst alert level across all experts. |
| `load_imbalance_score` | `float` | max_utilisation / mean_utilisation. 1.0 = perfect balance. `nan` when no data. |
| `is_empty` | `bool` | `True` when no routing events have been collected. |

**Properties:** `n_problematic`, `is_collapsed`, `is_degrading`.

**Methods:**

- `dead_experts() -> List[ExpertStatus]`
- `cold_experts() -> List[ExpertStatus]`
- `healthy_experts() -> List[ExpertStatus]`
- `expert(idx) -> ExpertStatus | None` — Return the `ExpertStatus` for expert index `idx`.
- `to_dict() -> dict`.

---

## 6. Collector layer

### `StatCollector`

```python
moewatch.collector.StatCollector(layer_names, config)
```

Aggregates `RoutingEvent` objects from all instrumented router layers into per-layer statistics. One `StatCollector` is shared by all `RouterHook` instances attached to the same model.

Maintains one `RingBuffer` per layer plus running accumulator totals for O(1) write and O(layers) read. Thread-safe for the single-writer (hook) / single-reader (analyzer) pattern.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `layer_names` | `List[str]` | Fully-qualified names of all router modules to track. |
| `config` | `WatchConfig` | Provides `ring_buffer_capacity` and `window_steps`. |

---

#### Methods

##### `StatCollector.add_event(event) -> None`

Integrate a single `RoutingEvent`. Called directly from `RouterHook` on every sampled forward pass. Never raises — errors are caught and logged so training continues unaffected. Auto-registers unknown layers rather than dropping data.

---

##### `StatCollector.get_layer_stats(layer_name) -> LayerStats | None`

Return aggregated `LayerStats` for a single layer, computed over the most-recent `config.window_steps` events. Returns `None` if the layer is unknown.

---

##### `StatCollector.get_all_stats() -> Dict[str, LayerStats]`

Return a snapshot of `LayerStats` for every tracked layer. Layers with no events are included with zero counts.

---

##### `StatCollector.clear(layer_name=None) -> None`

Clear collected events and running totals. If `layer_name` is provided, clears only that layer; otherwise clears all.

---

#### Properties

| Property | Type | Description |
|---|---|---|
| `tracked_layers` | `List[str]` | Names of all tracked layers in registration order. |
| `total_events` | `int` | Total events written across all layers (cumulative). |

**Methods:** `buffer_utilization() -> Dict[str, float]`, `summary() -> str`.

---

### `LayerStats`

```python
@dataclass
moewatch.collector.LayerStats
```

Aggregated routing statistics for a single MoE router layer. Produced by `StatCollector.get_layer_stats()` / `get_all_stats()` and consumed by the analyzer layer.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `layer_name` | `str` | Fully-qualified module name. |
| `n_experts` | `int` | Total number of experts. `0` if no events collected yet. |
| `expert_counts` | `torch.Tensor` | 1-D int64 CPU tensor of shape `(n_experts,)` — total tokens per expert over the rolling window. |
| `total_tokens` | `int` | Sum of `expert_counts`. |
| `utilization` | `torch.Tensor` | Float32 tensor of shape `(n_experts,)` — fraction of tokens per expert. All-zeros when `total_tokens == 0`. |
| `load_imbalance_score` | `float` | max_util / mean_util. Perfect balance → 1.0. `nan` when no data. |
| `top_k` | `int` | Number of experts each token was routed to. |
| `event_count` | `int` | Number of `RoutingEvent` objects in the window. |
| `step_range` | `Tuple[int, int]` | `(first_step, last_step)` of contributing events. `(-1, -1)` when empty. |
| `has_raw_logits` | `bool` | `True` if at least one event contained raw logit tensors. |
| `raw_logits_window` | `List[torch.Tensor]`, optional | Raw logit tensors from recent events. Each tensor: `(total_tokens_in_batch, n_experts)`. `None` when unavailable. |

**Properties:** `is_empty`.

**Methods:**

- `dead_mask(threshold=0.001) -> torch.Tensor` — Boolean mask, `True` for experts at or below `threshold` utilisation.
- `to_dict() -> dict`.

---

### `RingBuffer`

```python
moewatch.collector.RingBuffer(capacity)
```

Fixed-capacity circular buffer for `RoutingEvent` objects. Once full, the oldest entry is silently overwritten. O(1) append, O(n) snapshot. Thread-safe for single-writer / single-reader patterns.

**Parameters:** `capacity: int` — must be ≥ 1.

---

#### Methods

##### `RingBuffer.append(event) -> None`

Append a `RoutingEvent`. Raises `TypeError` for any other type. On first wrap, emits a DEBUG log suggesting to increase `ring_buffer_capacity`.

##### `RingBuffer.snapshot() -> List[RoutingEvent]`

Return a consistent point-in-time copy of all current events, oldest-first. Acquires a lock; safe against concurrent `clear()`.

##### `RingBuffer.latest(n=1) -> List[RoutingEvent]`

Return the `n` most-recent events, newest-first.

##### `RingBuffer.clear() -> None`

Remove all events without changing capacity. Thread-safe.

---

#### Properties

| Property | Type | Description |
|---|---|---|
| `capacity` | `int` | Maximum events the buffer can hold. |
| `is_full` | `bool` | `True` when buffer has reached capacity. |
| `has_wrapped` | `bool` | `True` if at least one overwrite has occurred. |
| `total_written` | `int` | Cumulative events written since creation (never resets). |
| `utilization` | `float` | Fraction of capacity currently used (0.0–1.0). |

`RingBuffer` supports `len()` and `iter()` (iterates a snapshot, safe against concurrent writes).

---

## 7. Hooks layer

### `HookManager`

```python
moewatch.hooks.HookManager(model, router_module_names, collector, config)
```

Owns the full lifecycle of all PyTorch forward hooks attached to router modules. Guarantees clean teardown in every code path.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `model` | `torch.nn.Module` | The model to instrument. Never modified. |
| `router_module_names` | `List[str]` | Fully-qualified module names to hook. |
| `collector` | `StatCollector` | Receives `RoutingEvent` objects from each hook. |
| `config` | `WatchConfig` | Forwarded to each `RouterHook`. |

---

#### Methods

##### `HookManager.attach() -> HookManager`

Register all router hooks on the model. Returns `self`. Idempotent — calling when already attached emits a `UserWarning` and returns.

**Raises:** `ValueError` if any name in `router_module_names` is not found in the model. Aborts fully — no partial attachment.

##### `HookManager.detach() -> HookManager`

Remove all registered hooks. Safe to call multiple times. Individual removal errors are logged and suppressed — guaranteed not to raise.

##### `HookManager.get_hook(layer_name) -> RouterHook | None`

Return the `RouterHook` for a given layer name, or `None`.

##### `HookManager.diagnostics() -> Dict[str, Dict[str, int]]`

Return per-hook call and error counts: `{layer_name: {"calls": int, "errors": int}}`.

---

#### Properties

| Property | Type | Description |
|---|---|---|
| `is_attached` | `bool` | `True` when hooks are currently active. |
| `hook_count` | `int` | Number of hooks currently registered. |

**Context manager:** `HookManager` implements `__enter__` / `__exit__`, calling `attach()` and `detach()`. Recommended pattern for `audit()`.

---

### `RouterHook`

```python
moewatch.hooks.RouterHook(layer_name, collector, config)
```

PyTorch `register_forward_hook` callback for a single MoE router module. One `RouterHook` is created per router module. Never raises inside a forward pass — extraction errors are caught, counted, and logged.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `layer_name` | `str` | Fully-qualified module name — used as the key in the collector. |
| `collector` | `StatCollector` | Receives the `RoutingEvent` after each instrumented pass. |
| `config` | `WatchConfig` | Provides `sample_every` for step-sampling. |

---

#### Extraction strategies (tried in order)

1. **Tuple/list output** — position `[1]` as `(total_tokens, n_experts)` logits (Mixtral style); position `[2]` as `(total_tokens, top_k)` indices (OLMoE style).
2. **Object with routing attributes** — `.router_logits` or `.expert_indices`.
3. **Single tensor** — interpreted as routing logits directly.
4. **Dict output** — keys `router_logits`, `gate_logits`, `routing_logits`, `expert_indices`, `selected_experts`, `top_k_indices`.

---

#### Properties

| Property | Type | Description |
|---|---|---|
| `call_count` | `int` | Total forward passes seen (before `sample_every` filter). |
| `error_count` | `int` | Non-fatal extraction errors encountered. |

---

### `RoutingEvent`

```python
@dataclass
moewatch.hooks.RoutingEvent
```

Immutable snapshot of a single router forward pass. Written by `RouterHook`, consumed by `StatCollector`.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `layer_name` | `str` | Fully-qualified module name. |
| `step` | `int` | Hook-internal call counter at capture time. |
| `timestamp` | `float` | Wall-clock capture time (seconds since epoch). |
| `n_experts` | `int` | Total number of experts in this layer. |
| `expert_counts` | `torch.Tensor` | 1-D int64 CPU tensor of shape `(n_experts,)` — token counts per expert. |
| `total_tokens` | `int` | Total tokens processed (sum of `expert_counts`; may exceed batch × seq_len for top-k > 1). |
| `raw_logits` | `torch.Tensor`, optional | CPU float32 tensor of shape `(total_tokens, n_experts)` — un-softmaxed router logits. `None` when logit extraction is not possible. |
| `top_k` | `int` | Number of experts each token was routed to. |

---

### `detect_router_modules()`

```python
moewatch.hooks.detect_router_modules(model) -> List[str]
```

Auto-detect MoE router module names in `model` using a two-pass strategy.

**Pass 1 — Architecture registry:** Matches against a curated registry of known class names for all supported architectures. Returns immediately on a registry hit.

**Pass 2 — Heuristic fallback:** Scans all module class names for substrings associated with MoE routing (`"router"`, `"gate"`, `"sparse"`, `"moe"`, etc.), filtered against known non-router classes (`"embedding"`, `"norm"`, `"attention"`, etc.). A minimum parameter count guard prevents false positives on tiny scalar modules.

**Parameters:** `model: torch.nn.Module` — must be fully initialised (weights loaded).

**Returns:** `List[str]` — fully-qualified router module names, deduplicated and ordered by depth (shallowest first). Empty list if detection fails (a detailed `WARNING` is logged).

**Companion functions:**

```python
from moewatch.hooks.detection import list_known_architectures, is_known_router_class

# Returns {family: [class_names]} — the full registry as a plain dict
list_known_architectures()

# True if class_name is in the curated registry
is_known_router_class("MixtralSparseMoeBlock")
```

---

## 8. Entropy utilities

Pure functions in `moewatch.analyzer.entropy`. Useful for custom analysis scripts.

### `compute_entropy()`

```python
moewatch.analyzer.entropy.compute_entropy(probs) -> float
```

Compute Shannon entropy H = -Σ pᵢ log₂(pᵢ) for a probability vector.

- Numerically safe: zero probabilities contribute zero (0·log(0) ≡ 0 by convention).
- Re-normalises the input if it does not sum to 1.0 within tolerance.
- Returns bits (log base 2). Range: [0.0, log₂(n_experts)].

**Parameters:** `probs: torch.Tensor` — 1-D float tensor. Must be non-negative.

**Returns:** `float`

**Raises:** `ValueError` if `probs` is not 1-D or contains negative values.

```python
from moewatch.analyzer.entropy import compute_entropy
import torch

compute_entropy(torch.tensor([0.5, 0.5]))          # 1.0 bit
compute_entropy(torch.tensor([1.0, 0.0, 0.0]))     # 0.0 bits (one-hot)
compute_entropy(torch.ones(8) / 8)                 # 3.0 bits (uniform over 8)
```

---

### `compute_entropy_from_logits()`

```python
moewatch.analyzer.entropy.compute_entropy_from_logits(logits) -> float
```

Compute routing entropy directly from raw router logit tensors. Applies softmax row-wise to obtain per-token routing distributions, then averages entropy across all tokens.

More accurate than count-based estimation for small batch sizes.

**Parameters:** `logits: torch.Tensor` — 2-D float tensor of shape `(total_tokens, n_experts)`.

**Returns:** `float` — mean per-token Shannon entropy in bits.

**Raises:** `ValueError` if `logits` is not 2-D or has < 1 expert.

---

### `max_entropy()`

```python
moewatch.analyzer.entropy.max_entropy(n_experts) -> float
```

Theoretical maximum Shannon entropy for `n_experts` (uniform distribution).

**Returns:** `log₂(n_experts)`. Returns `0.0` for `n_experts ≤ 1`.

---

### `normalised_entropy()`

```python
moewatch.analyzer.entropy.normalised_entropy(h, n_experts) -> float
```

Normalise absolute entropy to [0.0, 1.0] relative to the theoretical maximum.

**Returns:** `h / log₂(n_experts)`, clamped to [0.0, 1.0].

---

## 9. Report rendering

### `CLIReporter`

```python
moewatch.report.CLIReporter(config)
```

Renders a coloured ASCII diagnostic report from an `AuditReport` to stdout. Sole owner of all terminal output logic — `AuditReport` is a pure data object and never imports this class.

**Parameters:** `config: WatchConfig` — controls `no_color` flag.

Respects both `config.no_color` and the `NO_COLOR` environment variable (see [no-color.org](https://no-color.org/)).

---

#### Methods

##### `CLIReporter.print_summary(report) -> None`

Render the full diagnostic report to stdout. Renders six sections:

| Section | Content |
|---|---|
| A — Header | Logo, health banner (coloured by `OverallHealth`), run metadata table. |
| B — Entropy | Per-layer entropy table with bar indicators, alert level, and trend direction. |
| C — Utilization | Per-layer per-expert ASCII block histograms with imbalance score. |
| D — Dead experts | Compact table of all dead and cold experts with consecutive cold step counts. |
| E — Recommendations | Prioritised actionable fix suggestions. |
| F — Footer | Elapsed time, config digest, docs link. |

Any rendering error in a section is caught and a fallback message is printed — remaining sections continue unaffected.

---

**Notes on output format:**

- All output lines are capped at 80 characters to avoid wrapping on narrow terminals.
- Layer names longer than the column width are truncated from the left, preserving the most informative tail (e.g. `...31.block_sparse_moe`).
- Expert utilization bars use Unicode block elements (`▏▎▍▌▋▊▉█`) at 1/8-character precision.
- `OutputMode.SILENT` suppresses all output from this class.

---

## Notes on imports

All stable public symbols are re-exported from the top-level `moewatch` package:

```python
from moewatch import (
    audit,
    MoEWatch,
    MoEWatchCallback,
    Alert,
    WatchConfig,
    OutputMode,
    AlertLevel,
)
```

Internal sub-modules (`moewatch.analyzer`, `moewatch.collector`, `moewatch.hooks`, `moewatch.report`) are importable directly but are not part of the stable public API surface — their interfaces may change between minor versions.

---

*moewatch 0.1.0 · [GitHub](https://github.com/Abineshabee/moewatch) · Apache 2.0 · Built by [Abinesh](https://github.com/Abineshabee)*
