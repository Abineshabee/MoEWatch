# Getting Started with MoEWatch

> **The pytest for Mixture-of-Experts models.** Catch expert collapse, routing entropy collapse, and load imbalance — before they silently wreck your training run.

---

## ⚡ Quick Installation

```bash
pip install moewatch
```

**Requirements:** Python ≥3.8, PyTorch ≥2.0, Transformers ≥4.36

---

## 🚀 Three Ways to Use MoEWatch

### 1️⃣ One-Shot Offline Audit (Recommended for Quick Checks)

Run diagnostics on a trained or pre-trained model without modifying your training code:

```python
import moewatch

# Your MoE model (Mixtral, OLMoE, DeepSeek-MoE, etc.)
model = ...
dataloader = ...

# Run 200-step diagnostic
report = moewatch.audit(model, dataloader, steps=200)

# Get human-readable summary
print(report.summary())
```

**When to use:** 
- Before starting a training run
- After training completes
- For CI/CD pipelines
- Quick sanity checks

---

### 2️⃣ Live Monitoring with HuggingFace Trainer (Easiest for Training)

One-line integration with your existing Trainer:

```python
from moewatch import MoEWatch

watcher = MoEWatch(model)
watcher.attach(trainer)  # ← that's it!

trainer.train()

watcher.detach()
```

MoEWatch automatically:
- ✅ Detects router modules
- ✅ Hooks into the training loop
- ✅ Emits alerts to console
- ✅ Cleans up on completion

---

### 3️⃣ Live Monitoring with Custom Training Loop (Full Control)

For non-Trainer training workflows:

```python
from moewatch import MoEWatch

watcher = MoEWatch(model)
watcher.start()  # attach hooks

for step, batch in enumerate(dataloader):
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    
    # Check health every log_every steps
    alerts = watcher.step(step)
    
    # Use alerts if needed
    if alerts:
        for alert in alerts:
            print(f"[{alert.level}] {alert.message}")

watcher.stop()  # clean up hooks
```

---

## 📊 Understanding Alerts

MoEWatch emits three severity levels:

| Level | Meaning | Action |
|-------|---------|--------|
| **INFO** ✅ | Everything healthy | Monitor normally |
| **WARN** ⚠️ | Early degradation | Investigate soon |
| **ERROR** ❌ | Severe collapse | Immediate attention |

### Alert Types

**Expert Collapse** — An expert receives <0.1% of tokens
```
❌ [Step 1000] ERROR  Expert 3 DEAD — utilisation: 0.002%
   💡 Suggestion: Raise aux_loss_coef or lower router learning rate
```

**Routing Entropy** — Router distribution is severely concentrated
```
⚠️  [Step 500] WARN   Routing entropy LOW: 45% of max entropy
```

**Load Imbalance** — One expert dominates token dispatch
```
❌ [Step 750] ERROR  Load imbalance CRITICAL: max/mean = 6.2×
   💡 Suggestion: Add capacity factor or switch to expert-choice routing
```

---

## ⚙️ Basic Configuration

### Default (Recommended Starting Point)

```python
from moewatch import WatchConfig, MoEWatch

config = WatchConfig()
watcher = MoEWatch(model, config=config)
```

### Three Built-in Presets

```python
# Debugging mode — catches everything (4–8% overhead)
config = WatchConfig.aggressive()

# Production mode — minimal overhead (<2%)
config = WatchConfig.lightweight()

# Silent mode — no real-time output
config = WatchConfig.silent()
```

### Common Customizations

```python
config = WatchConfig(
    # Expert thresholds
    dead_threshold=0.001,      # < 0.1% utilisation → DEAD
    cold_threshold=0.005,      # < 0.5% utilisation → COLD
    
    # Entropy thresholds
    entropy_warn=0.60,         # < 60% of max → WARN
    entropy_critical=0.40,     # < 40% of max → ERROR
    
    # Load imbalance
    load_imbalance_error=5.0,  # max/mean > 5× → ERROR
    
    # Sampling (tune for overhead vs fidelity)
    sample_every=10,           # check every 10th forward pass
    log_every=100,             # emit alerts every 100 steps
    
    # Output mode
    output="console",          # "console" | "json" | "silent"
)

watcher = MoEWatch(model, config=config)
```

---

## 🎯 Common Workflows

### Before Training: Audit a Pre-trained Model

```python
from transformers import AutoModelForCausalLM
import moewatch

model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
report = moewatch.audit(model, steps=100)

if report.has_collapse:
    print(f"⚠️  {len(report.dead_experts())} dead experts detected")
else:
    print("✅ Model looks healthy. Safe to train.")
```

### During Training: Monitor in Real-time

```python
from transformers import Trainer
from moewatch import MoEWatch

trainer = Trainer(model=model, args=training_args, ...)
watcher = MoEWatch(model)
watcher.attach(trainer)

trainer.train()
watcher.detach()

# Access alert history
for alert in watcher.get_alert_log():
    if alert.level.value == "ERROR":
        print(f"Caught error at step {alert.step}: {alert.message}")
```

### Custom Loop with JSON Output (for log aggregation)

```python
from moewatch import MoEWatch, WatchConfig

config = WatchConfig(output="json")  # Newline-delimited JSON
watcher = MoEWatch(model, config=config)
watcher.start()

for step, batch in enumerate(dataloader):
    # ... training code ...
    watcher.step(step)

watcher.stop()
```

Parse the JSON output with your log pipeline:
```bash
python train.py 2>&1 | grep '{"step"' | jq '.message'
```

---

## 🔧 Advanced: Custom Router Architecture

If MoEWatch doesn't auto-detect your custom MoE architecture:

```python
from moewatch import WatchConfig, MoEWatch

# Specify router module names explicitly
config = WatchConfig(
    router_modules=[
        "model.layers.0.moe_router",
        "model.layers.1.moe_router",
        # ... all your router layers
    ]
)

watcher = MoEWatch(model, config=config)
watcher.start()
```

**How to find your router modules:**
```python
for name, module in model.named_modules():
    if "router" in name.lower() or "gate" in name.lower():
        print(name)
```

---

## 📈 Interpreting the Report

After running `audit()`, you get an `AuditReport` with:

```python
report = moewatch.audit(model, dataloader, steps=200)

# Summary health
print(report.has_collapse)              # bool
print(report.overall_health)            # OverallHealth enum

# Dead experts
dead = report.dead_experts()            # List[DeadExpertEntry]
for expert in dead:
    print(f"Layer {expert.layer}: expert {expert.expert_id} "
          f"({expert.utilization:.2%} utilisation)")

# Entropy trends
print(report.entropy_summary)           # Per-layer entropy stats

# Load imbalance
print(report.imbalance_summary)         # Per-layer imbalance scores

# Render to console
print(report.summary())                 # Pretty-printed report
```

---

## ❓ Troubleshooting

### Q: "Could not detect any router modules"

**Solution:** Specify them manually:
```python
config = WatchConfig(router_modules=["your.router.path"])
```

### Q: "Overhead is too high"

**Solution:** Use lightweight mode:
```python
config = WatchConfig.lightweight()  # sample_every=50
```

### Q: "I want silent mode for CI"

**Solution:** Use silent config and inspect the report:
```python
config = WatchConfig.silent()
report = moewatch.audit(model, config=config)
assert not report.has_collapse, "Expert collapse detected!"
```

### Q: "How do I integrate with wandb?"

**Solution:** Parse alerts and log them:
```python
import wandb

watcher = MoEWatch(model)
watcher.start()

for step in range(num_steps):
    alerts = watcher.step(step)
    for alert in alerts:
        wandb.log({"alert_level": alert.level.value, 
                   "message": alert.message})

watcher.stop()
```

---

## 🔗 Supported Architectures

**Auto-detected** (no configuration needed):

- ✅ Mixtral (mistralai)
- ✅ OLMoE (allenai)
- ✅ DeepSeek-MoE (deepseek-ai)
- ✅ Qwen-MoE (Alibaba)
- ✅ Phi-MoE (Microsoft)
- ✅ Switch Transformer (Google)
- ✅ NLLB-MoE (Meta)
- ✅ Arctic (Snowflake)
- ✅ Jamba (AI21 Labs)

Not in the list? Use manual `router_modules` override.

---

## 📚 Next Steps

- Read [**Configuration Reference**](./config.md) for all tunable parameters
- Check [**API Reference**](./api_reference.md) for detailed method docs
- See [CONTRIBUTING.md](../CONTRIBUTING.md) to extend MoEWatch

---

**Questions?** Open an issue on [GitHub](https://github.com/Abineshabee/moewatch) or email abineshabee2@gmail.com
