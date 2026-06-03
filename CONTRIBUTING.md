# Contributing to moewatch

> Thanks for your interest in contributing! moewatch is a focused diagnostic library for MoE models — we keep the scope tight and the code clean.

---

## 📁 Project Structure

> [!NOTE]
> View the interactive project structure explorer here:
> **[🗂️ Browse Project Structure](https://htmlpreview.github.io/?https://github.com/Abineshabee/MoEWatch/blob/main/docs/moewatch_project_structure.html)**

```
moewatch/
├── moewatch/               # Installable package
│   ├── _audit.py           # audit() — offline diagnostic entry point
│   ├── _watcher.py         # MoEWatch — live training monitor
│   ├── config.py           # WatchConfig — all thresholds and settings
│   ├── hooks/              # PyTorch forward hook machinery
│   ├── collector/          # Async stats aggregation (ring buffer)
│   ├── analyzer/           # Entropy + collapse metric computation
│   └── report/             # AuditReport data object + CLIReporter
├── tests/                  # Full CPU test suite (no GPU required)
├── examples/               # End-to-end integration examples
├── benchmarks/             # Overhead benchmarks vs unmonitored baseline
├── docs/                   # API reference + project explorer
└── .github/workflows/      # CI (lint → typecheck → test) + PyPI publish
```

---

## ⚡ Quick Start

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/MoEWatch.git
cd MoEWatch/moewatch

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Verify everything passes
ruff check moewatch/
mypy moewatch/
pytest
```

All tests run on CPU — no GPU or model weights required.

---

## ✅ Before You Open a PR

Every PR must pass the full CI pipeline locally:

| Check | Command | What it enforces |
|---|---|---|
| Lint | `ruff check moewatch/` | Style and unused imports |
| Format | `black --check moewatch/` | Code formatting (run `black moewatch/` to auto-fix) |
| Types | `mypy moewatch/` | Type correctness |
| Tests | `pytest` | 80% coverage minimum |

---

## 🧩 How the Codebase Fits Together

```
audit() / MoEWatch.step()
        │
        ▼
   HookManager  ── attaches ──▶  RouterHook (per router module)
                                      │
                                      ▼ 
                                RoutingEvent
                                StatCollector (ring buffer)
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
             EntropyAnalyzer                      CollapseDetector
                    │                                    │
                    └─────────────────┬──────────────────┘
                                      ▼
                                 AuditReport
                                      │
                                      ▼
                                 CLIReporter
```

**Key principle:** hooks are read-only — the model is never modified. All metric computation is pure functions in `analyzer/` with no side effects.

---

## 🔧 Common Contribution Areas

### Adding a new router architecture

1. Open `moewatch/hooks/detection.py`
2. Add the class name to the architecture registry in `_KNOWN_ROUTER_CLASSES`
3. Add a test in `tests/test_detection.py` with a mock module using that class name

### Adding a new metric

1. Add a pure function to `moewatch/analyzer/entropy.py` or `collapse.py`
2. Wire it into `EntropyAnalyzer` or `CollapseDetector`
3. Expose the result via `AuditReport` in `moewatch/report/audit_report.py`
4. Cover it in the corresponding test file

### Adding a new WatchConfig threshold

1. Add the field to `WatchConfig` in `moewatch/config.py`
2. Add validation in `__post_init__`
3. Update `to_dict()` and `__repr__`

---

## 🧪 Writing Tests

Tests live in `tests/` and use fixtures from `conftest.py`:

```python
def test_my_feature(synthetic_model, tiny_dataloader, default_config):
    # synthetic_model  — minimal MoE, CPU only, no real weights
    # tiny_dataloader  — random tensors, batch size 2
    # default_config   — WatchConfig with safe defaults
    ...
```

- **No GPU required** — all tests must pass on CPU
- **No real model weights** — use `synthetic_model` fixture
- Keep tests focused — one behaviour per test function

---

## 📐 Code Style

- **Formatter:** `black` (line length 88)
- **Linter:** `ruff`
- **Types:** all public functions need type annotations
- **Docstrings:** Google style for public API; internal helpers can be brief
- **No model modification** — hooks are read-only, always

---

## 🚀 Submitting a PR

1. Branch off `main`: `git checkout -b feat/your-feature`
2. Make your changes and ensure CI passes locally
3. Open a PR against `main` with a clear description of what and why
4. Link any related issue

---

## 📬 Reporting Issues

Use the GitHub issue templates:

- **[🐛 Bug Report](https://github.com/Abineshabee/MoEWatch/issues/new?template=bug_report.md)** — include moewatch version, model name, Python version, and a minimal repro
- **[✨ Feature Request](https://github.com/Abineshabee/MoEWatch/issues/new?template=feature_request.md)** — describe the use case and target architecture

---

*moewatch 0.1.0 · Apache 2.0 · [API Reference](docs/api_reference.md)*
