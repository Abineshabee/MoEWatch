# Configuration Reference

> All tunable thresholds, sampling rates, and output modes for MoEWatch diagnostics.

---

## 🎯 Overview

MoEWatch behavior is controlled entirely through a single **immutable** configuration object: `WatchConfig`. Pass it once to `audit()`, `MoEWatch()`, or any sub-component — there is no global mutable state.

```python
from moewatch import WatchConfig, MoEWatch

config = WatchConfig(
    dead_threshold=0.001,
    entropy_warn=0.60,
    output="console",
)

watcher = MoEWatch(model, config=config)
```

---

## ⚡ Quick Start Presets

Use these for 90% of cases. Only customize beyond if you have specific needs.

### Default (Balanced)

```python
config = WatchConfig()
```

| Setting | Value | Notes |
|---------|-------|-------|
| Overhead | ~2% | Balanced for production |
| Sampling | Every 10 steps | Good fidelity |
| Thresholds | Literature defaults | Works for most models |

---

### Aggressive (Debugging)

```python
config = WatchConfig.aggressive()
```

| Setting | Value | Notes |
|---------|-------|-------|
| Overhead | 4–8% | High fidelity for short runs |
| Sampling | Every step | Captures all routing events |
| Thresholds | Tighter | Catches early degradation |
| Window | Shorter | Faster trend detection |

**When to use:** Debugging expert collapse in short experimental runs.

---

### Lightweight (Production)

```python
config = WatchConfig.lightweight()
```

| Setting | Value | Notes |
|---------|-------|-------|
| Overhead | <1% | Minimal impact on training |
| Sampling | Every 50 steps | Statistical estimates still valid |
| Buffer | Compact | Lower memory footprint |
| Logging | Less frequent | 500 steps between alerts |

**When to use:** Large-scale production training (100B+ tokens).

---

### Silent (CI/CD)

```python
config = WatchConfig.silent()
```

| Setting | Value | Notes |
|---------|-------|-------|
| Output | None | No console/JSON output |
| Results | Via `AuditReport` object | Inspect programmatically |

**When to use:** Automated testing, CI pipelines, gated training.

---

## 📋 Complete Parameter Reference

### Expert Health Thresholds

These control when an expert is flagged as **COLD** or **DEAD**.

#### `dead_threshold` — float

**Default:** `0.001` (0.1%)

Fraction of total tokens below which an expert is considered **permanently dead**. 

```python
config = WatchConfig(dead_threshold=0.001)
```

| Model Scale | Typical Value | Reasoning |
|---|---|---|
| Small (8 experts) | `0.001` | Tighter: fewer experts to lose |
| Medium (32 experts) | `0.0005` | More recovery headroom |
| Large (64+ experts) | `0.0001` | Many redundant experts |

> **Based on:** Switch Transformer training logs show experts below 0.1% rarely recover.

---

#### `cold_threshold` — float

**Default:** `0.005` (0.5%)

Fraction of total tokens below which an expert is considered **potentially recoverable** (COLD). Must be **strictly greater** than `dead_threshold`.

```python
config = WatchConfig(
    dead_threshold=0.001,
    cold_threshold=0.005,  # must be > dead_threshold
)
```

| Setting | Meaning |
|---------|---------|
| `utilization < dead_threshold` | DEAD (unrecoverable) |
| `dead_threshold ≤ util < cold_threshold` | COLD (early warning) |
| `utilization ≥ cold_threshold` | HEALTHY (normal) |

---

#### `cold_steps_limit` — int

**Default:** `500`

Number of **consecutive steps** an expert must remain COLD before being promoted to DEAD status.

```python
config = WatchConfig(cold_steps_limit=500)
```

**Example:** If expert stays COLD for 500 consecutive analysis steps, it's flagged ERROR.

| Training Scenario | Suggested Value |
|---|---|
| Short fine-tuning (<10K steps) | `100–200` |
| Standard training (10K–100K) | `500` (default) |
| Long pretraining (>1M steps) | `1000–2000` |

---

### Routing Entropy Thresholds

Entropy is the **primary early-warning signal** for collapse. A healthy router distributes tokens near-uniformly; entropy close to **log₂(n_experts)** is ideal.

#### `entropy_warn` — float

**Default:** `0.60`

Fraction of **theoretical maximum entropy** below which a WARNING is emitted.

```python
config = WatchConfig(entropy_warn=0.60)
```

**Example (8 experts):**
- Theoretical max: H_max = log₂(8) = 3.0 bits
- WARN threshold: 0.60 × 3.0 = 1.8 bits
- If measured entropy < 1.8 bits → ⚠️ WARNING

| Model Type | Suggested Value |
|---|---|
| Dense (baseline) | `0.70` → more lenient |
| Balanced MoE | `0.60` (default) |
| Aggressive routing | `0.50` → tighter |

---

#### `entropy_critical` — float

**Default:** `0.40`

Fraction of theoretical maximum entropy below which an ERROR is emitted. Must be **strictly less** than `entropy_warn`.

```python
config = WatchConfig(
    entropy_warn=0.60,
    entropy_critical=0.40,  # must be < entropy_warn
)
```

**Example (8 experts):**
- ERROR threshold: 0.40 × 3.0 = 1.2 bits
- If measured entropy < 1.2 bits → ❌ ERROR

---

#### `entropy_drop_warn` — float

**Default:** `0.20`

Relative entropy **drop over the rolling window** that triggers a WARNING. Detects monotonic degradation even before absolute thresholds are crossed.

```python
config = WatchConfig(entropy_drop_warn=0.20)
```

**Example:**
- Rolling window entropy: [3.0, 2.9, 2.8, 2.5]
- Drop: (3.0 − 2.5) / 3.0 = 16.7%
- Threshold: 20%
- Result: ✅ Still OK (no warn)

| Use Case | Suggested Value |
|---|---|
| Stable models | `0.15` (sensitive) |
| Standard training | `0.20` (default) |
| Noisy training | `0.30` (lenient) |

---

### Load Imbalance Thresholds

Measures how skewed the token distribution is across experts: **max expert load / mean load**.

#### `load_imbalance_warn` — float

**Default:** `3.0`

Ratio of max expert utilisation to mean utilisation above which a WARNING is emitted.

```python
config = WatchConfig(load_imbalance_warn=3.0)
```

**Example (8 experts, uniform load):**
- Mean load: 12.5% per expert
- If one expert gets 40%: ratio = 40 / 12.5 = 3.2× → ⚠️ WARN

| Scenario | Threshold | Note |
|---|---|---|
| Strict routing | `2.5` | Nearly uniform expected |
| Token-choice (top-2) | `3.0` (default) | Some imbalance OK |
| Top-k with capacity | `4.0` | More variance acceptable |

---

#### `load_imbalance_error` — float

**Default:** `5.0`

Ratio above which an ERROR is emitted. Must be **strictly greater** than `load_imbalance_warn`.

```python
config = WatchConfig(
    load_imbalance_warn=3.0,
    load_imbalance_error=5.0,  # must be > warn
)
```

| Threshold | Severity |
|---|---|
| < `load_imbalance_warn` | ✅ OK |
| `warn` to `error` | ⚠️ WARN |
| > `load_imbalance_error` | ❌ ERROR |

---

### Sampling & Windowing

Control the trade-off between **monitoring fidelity** and **training overhead**.

#### `sample_every` — int

**Default:** `10`

Instrument **every Nth forward pass**. Higher values → lower overhead, but coarser data.

```python
config = WatchConfig(sample_every=10)
```

| Value | Overhead | Use Case |
|---|---|---|
| `1` | 2–4% | Debugging collapse (⚠️ not for production) |
| `5` | 1–2% | Balanced debugging |
| `10` | <1% | Production (default) |
| `50` | <0.5% | Ultra-lightweight production |

> **Note:** Even with `sample_every=50`, the sliding window analysis gives statistically valid estimates of expert health.

---

#### `log_every` — int

**Default:** `100`

Emit alert summaries **every Nth training step** (independent of `sample_every`). Alerts fire only when there's something to report.

```python
config = WatchConfig(log_every=100)
```

**Example:**
- Training loop runs every step
- Sampling happens every 10 steps
- Alerts emit only at steps 0, 100, 200, etc.

| Training Length | Suggested Value |
|---|---|
| Short run (<5K steps) | `50` |
| Standard training (10K–100K) | `100` (default) |
| Long training (>1M steps) | `500` |

---

#### `window_steps` — int

**Default:** `500`

Rolling **window size** (in training steps) for trend analysis and expert health tracking.

```python
config = WatchConfig(window_steps=500)
```

**What it does:**
- Entropy trend detection uses stats from the last N steps
- Cold-expert duration counter resets when expert recovers within the window

| Window Size | Meaning |
|---|---|
| Small (100) | Highly reactive, noisy trends |
| Default (500) | Good balance |
| Large (2000) | Stable long-term trends |

---

#### `ring_buffer_capacity` — int

**Default:** `10_000`

Maximum number of **RoutingEvent entries** held in memory. Older events are discarded when the buffer wraps.

```python
config = WatchConfig(ring_buffer_capacity=10_000)
```

| Memory Budget | Capacity | Est. Size |
|---|---|---|
| Tight (<1 GB) | `2_000` | ~10 MB |
| Standard | `10_000` (default) | ~50 MB |
| Unlimited | `50_000` | ~250 MB |

> Tune this if you run **very long training** with many layers/experts.

---

### Output & Display

Control how diagnostics are presented.

#### `output` — OutputMode or str

**Default:** `OutputMode.CONSOLE`

How moewatch emits real-time diagnostic output.

```python
from moewatch import OutputMode

# Option 1: Enum
config = WatchConfig(output=OutputMode.JSON)

# Option 2: String (auto-coerced)
config = WatchConfig(output="json")
```

| Mode | Output | Use Case |
|---|---|---|
| `"console"` | Coloured ASCII to stdout | Interactive training, notebooks |
| `"json"` | Newline-delimited JSON | Log aggregators (Grafana, Splunk) |
| `"silent"` | Nothing | CI/CD, programmatic access only |

**Example JSON output:**
```json
{"step": 1000, "level": "ERROR", "layer_name": "layers.5.moe", "message": "Expert 3 DEAD", "metric": 0.0008}
```

---

#### `no_color` — bool

**Default:** `False`

Disable ANSI colour codes in console output. Automatically respects the `NO_COLOR` environment variable.

```python
# Programmatic override
config = WatchConfig(no_color=True)

# Or use env var
export NO_COLOR=1
python train.py
```

**When to use:**
- CI/CD logs (no terminal support)
- Log files (plain text preferred)
- Accessibility requirements

---

### Router Discovery

#### `router_modules` — list of str

**Default:** `[]` (empty — auto-detect)

Explicit list of fully-qualified router module names. When provided, bypasses auto-detection entirely.

```python
config = WatchConfig(
    router_modules=[
        "model.layers.0.moe.router",
        "model.layers.1.moe.router",
        # ... all your routers
    ]
)
```

**When to use:**
- Custom architectures not in the auto-detection registry
- You want explicit control over which modules are monitored
- Models with unusual naming schemes

**How to find your routers:**
```python
for name, module in model.named_modules():
    if "router" in name.lower() or "gate" in name.lower():
        print(name)
```

---

## 🔗 Creating Custom Configurations

### Example: Debugging Expert Collapse

```python
config = WatchConfig(
    # Catch dead experts early
    dead_threshold=0.0005,      # even lower bar
    cold_threshold=0.002,
    cold_steps_limit=100,       # quick promotion to DEAD
    
    # Sensitive entropy checks
    entropy_warn=0.70,
    entropy_critical=0.50,
    entropy_drop_warn=0.10,
    
    # High-fidelity sampling
    sample_every=1,             # every step
    log_every=50,               # frequent alerts
    window_steps=200,           # responsive trends
    
    # Console output for interactive debugging
    output="console",
)

watcher = MoEWatch(model, config=config)
watcher.start()
```

---

### Example: Production Training (Minimal Overhead)

```python
config = WatchConfig(
    # Standard thresholds (no change)
    
    # Minimal sampling
    sample_every=100,           # every 100 steps
    log_every=1000,             # log every 1000 steps
    ring_buffer_capacity=2000,  # compact buffer
    
    # JSON for log pipeline
    output="json",
)

watcher = MoEWatch(model, config=config)
watcher.start()
```

---

### Example: CI/CD Gating

```python
config = WatchConfig(
    # Standard health thresholds
    dead_threshold=0.001,
    entropy_critical=0.40,
    load_imbalance_error=5.0,
    
    # Silent — results via AuditReport only
    output="silent",
)

report = moewatch.audit(model, dataloader, config=config)

# Gate training on health
if report.has_collapse:
    print(f"FAILED: {len(report.dead_experts())} dead experts")
    exit(1)
else:
    print("PASSED: Model is healthy")
    exit(0)
```

---

## ⚠️ Common Mistakes

### ❌ Mistake 1: `dead_threshold ≥ cold_threshold`

```python
# INVALID
config = WatchConfig(
    dead_threshold=0.005,
    cold_threshold=0.005,  # Must be GREATER than dead_threshold!
)
```

**Fix:**
```python
config = WatchConfig(
    dead_threshold=0.001,
    cold_threshold=0.005,  # ✅ Correct
)
```

---

### ❌ Mistake 2: `entropy_critical ≥ entropy_warn`

```python
# INVALID
config = WatchConfig(
    entropy_warn=0.50,
    entropy_critical=0.50,  # Must be LESS than warn!
)
```

**Fix:**
```python
config = WatchConfig(
    entropy_warn=0.60,
    entropy_critical=0.40,  # ✅ Correct (40 < 60)
)
```

---

### ❌ Mistake 3: `load_imbalance_warn ≥ load_imbalance_error`

```python
# INVALID
config = WatchConfig(
    load_imbalance_warn=5.0,
    load_imbalance_error=5.0,  # Must be GREATER than warn!
)
```

**Fix:**
```python
config = WatchConfig(
    load_imbalance_warn=3.0,
    load_imbalance_error=5.0,  # ✅ Correct (5 > 3)
)
```

---

## 🔍 Serialization

Export your config to a dictionary for logging or reproduction:

```python
config = WatchConfig(dead_threshold=0.001, entropy_warn=0.60)

# Get JSON-serializable dict
config_dict = config.to_dict()
print(config_dict)

# All fields included, ready for logging
{
    "dead_threshold": 0.001,
    "entropy_warn": 0.60,
    "entropy_critical": 0.4,
    ...
}
```

---

## 📚 Default Values Summary

| Parameter | Default | Range |
|---|---|---|
| `dead_threshold` | `0.001` | (0, 1) |
| `cold_threshold` | `0.005` | (0, 1) |
| `cold_steps_limit` | `500` | ≥ 1 |
| `entropy_warn` | `0.60` | (0, 1) |
| `entropy_critical` | `0.40` | (0, 1) |
| `entropy_drop_warn` | `0.20` | (0, 1) |
| `load_imbalance_warn` | `3.0` | > 1 |
| `load_imbalance_error` | `5.0` | > 1 |
| `window_steps` | `500` | ≥ 1 |
| `sample_every` | `10` | ≥ 1 |
| `log_every` | `100` | ≥ 1 |
| `ring_buffer_capacity` | `10_000` | ≥ 1 |
| `output` | `"console"` | — |
| `no_color` | `False` | — |
| `router_modules` | `[]` | — |

---

## 🎓 Further Reading

- [Getting Started](./quickstart.md) — Three ways to use MoEWatch
- [API Reference](./api_reference.md) — Detailed method documentation
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Extending the library

---

**Questions?** Open an [issue on GitHub](https://github.com/Abineshabee/moewatch) or email abineshabee2@gmail.com.
