
<p align="center">
  <img src="assets/images/banner.svg" width="80%" />
</p>

> **The pytest for Mixture-of-Experts models.** Catch expert collapse, routing entropy collapse, and load imbalance — before they silently wreck your training run.

[![PyPI version](https://img.shields.io/pypi/v/moewatch.svg)](https://pypi.org/project/moewatch/)
[![Python](https://img.shields.io/pypi/pyversions/moewatch.svg)](https://pypi.org/project/moewatch/)
[![CI](https://github.com/Abineshabee/MoEWatch/actions/workflows/ci.yml/badge.svg)](https://github.com/Abineshabee/MoEWatch/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-Abineshabee%2Fmoewatch-black?logo=github)](https://github.com/Abineshabee/moewatch)

---

MoEWatch is a lightweight diagnostic and audit library for MoE models in HuggingFace Transformers. Drop it into any training loop — it instruments router modules with zero-weight-modification PyTorch hooks, aggregates routing statistics, and surfaces structured alerts the moment something goes wrong.

---

## Features

- **Expert collapse detection** — tracks dead and cold experts per layer across the full training run
- **Routing entropy analysis** — catches distribution collapse relative to theoretical maximum entropy
- **Load imbalance alerts** — fires when any single expert dominates token dispatch (max/mean ratio)
- **Auto-detection** — recognises Mixtral, OLMoE, DeepSeek-MoE, Qwen-MoE, Phi-MoE, Switch Transformer, and more out of the box; falls back to heuristic scan for unknown architectures
- **Two integration modes** — one-shot `audit()` for offline diagnostics, or `MoEWatch` for live training-time monitoring
- **HuggingFace `Trainer` support** — attach as a `TrainerCallback` with one line
- **Structured output** — console (coloured ASCII), JSON (for log pipelines), or silent (results only via `AuditReport`)
- **Configurable overhead** — `sample_every=10` keeps instrumentation below 2 % in production; `sample_every=1` for maximum fidelity during debugging
- **Fixed memory footprint** — ring buffer with configurable capacity; no unbounded growth over long runs

---

## Supported Architectures

Auto-detected via registry (no configuration needed):

| Family | Models |
|---|---|
| Mixtral | `mistralai/Mixtral-*` |
| OLMoE | `allenai/OLMoE-*` |
| DeepSeek-MoE | `deepseek-ai/DeepSeek-V2`, `DeepSeek-V3` |
| Qwen-MoE | `Qwen/Qwen2-MoE-*`, `Qwen3-MoE-*` |
| Phi-MoE | `microsoft/Phi-*-MoE` |
| Switch Transformer | Google's HuggingFace port |
| NLLB-MoE | `facebook/nllb-moe-*` |
| Arctic | `Snowflake/snowflake-arctic-*` |
| Jamba | `ai21labs/Jamba-*` |

Any custom architecture can be targeted via `WatchConfig(router_modules=[...])`.

---

## Installation

```bash
pip install moewatch
```

Requires Python ≥ 3.8, PyTorch ≥ 1.10, and Transformers (optional — required only for `MoEWatch.attach(trainer)`).

---

## Quick Start

### Offline audit (one-shot)

Run a diagnostic against a model and dataloader without modifying your training loop:

```python
import moewatch

report = moewatch.audit(model, dataloader, steps=200)
print(report.summary())
```

### Live monitoring (HuggingFace Trainer)

```python
from moewatch import MoEWatch, WatchConfig

watcher = MoEWatch(model, config=WatchConfig())
watcher.attach(trainer)          # injects as a TrainerCallback
trainer.train()
watcher.detach()
```

### Live monitoring (custom loop)

```python
from moewatch import MoEWatch

watcher = MoEWatch(model)
watcher.start()

for step, batch in enumerate(dataloader):
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    alerts = watcher.step(step)   # returns List[Alert]; empty when healthy

watcher.stop()
```

---

## Configuration

All thresholds and options live in `WatchConfig`. Three presets cover most use cases:

```python
from moewatch import WatchConfig

WatchConfig.default()      # balanced — recommended starting point
WatchConfig.aggressive()   # tighter thresholds, every-step sampling — for debugging
WatchConfig.lightweight()  # minimal overhead — for large-scale production runs
```

Common overrides:

```python
config = WatchConfig(
    dead_threshold=0.001,        # < 0.1 % token share → expert is DEAD
    entropy_warn=0.60,           # < 60 % of H_max → WARN
    entropy_critical=0.40,       # < 40 % of H_max → ERROR
    load_imbalance_error=5.0,    # max/mean > 5× → ERROR
    sample_every=10,             # instrument every 10th forward pass
    output="json",               # "console" | "json" | "silent"
)
```

See the [Configuration reference →](https://github.com/Abineshabee/moewatch/docs/config.md) for all fields and their defaults.

---

## Alert Levels

| Level | Meaning |
|---|---|
| `INFO` | Routine routing statistics — everything healthy |
| `WARN` | Degraded routing — investigate soon |
| `ERROR` | Severe collapse or imbalance — likely harming training |

MoEWatch never stops your training run. It diagnoses; you decide.

---

## Output Modes

```python
# Human-readable console output (default)
WatchConfig(output="console")

# Newline-delimited JSON — pipe to Grafana, Splunk, or a custom pipeline
WatchConfig(output="json")

# No real-time output — results available only via AuditReport
WatchConfig(output="silent")
```

---

## Documentation

- [Getting started](https://github.com/Abineshabee/MoEWatch/blob/main/docs/quickstart.md)
- [Configuration reference](https://github.com/Abineshabee/MoEWatch/blob/main/docs/config.md)
- [API reference](https://github.com/Abineshabee/MoEWatch/blob/main/docs/api_reference.md)
- [Adding a custom architecture](https://github.com/Abineshabee/MoEWatch/blob/main/docs/custom-architecture.md)
- [Contributing](CONTRIBUTING.md)

---

## Contributing

Issues and pull requests are welcome. To add a new architecture to the auto-detection registry, open an issue or add the router class name(s) to `_ARCHITECTURE_REGISTRY` in `hooks/detection.py` and submit a PR.

For full contribution guidelines, see [**CONTRIBUTING.md**](CONTRIBUTING.md).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Built by [Abinesh](https://github.com/Abineshabee).*
