# Adding a Custom Architecture

> Extend MoEWatch to monitor your custom Mixture-of-Experts architecture.

---

## 🎯 Overview

MoEWatch ships with **auto-detection** for 9 popular MoE architectures (Mixtral, OLMoE, DeepSeek, etc.). If your model isn't in that list, you have **two options:**

| Approach | Effort | Persistence | Best For |
|----------|--------|-------------|----------|
| **Quick fix** | 5 min | This session only | One-off experiments |
| **Permanent** | 15 min | Forever (all users) | Custom production models |

This guide covers both.

---

## 🔍 Step 1: Identify Your Router Modules

Before you can monitor a model, you need to find where the routing decisions happen.

### Quick Discovery Script

Run this to list all modules in your model:

```python
import torch.nn as nn

def find_routers(model):
    """Print all modules with 'router' or 'gate' in the name."""
    print("Potential router modules:")
    for name, module in model.named_modules():
        class_name = type(module).__name__
        
        # Flag potential routers
        if any(keyword in name.lower() for keyword in ["router", "gate", "sparse", "moe"]):
            print(f"  ✅ {name}")
            print(f"     Class: {class_name}")
            print()

# Usage
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your/model")
find_routers(model)
```

### What You're Looking For

A **router module** is where the MoE gating happens. It typically:

1. **Takes token embeddings as input**
2. **Outputs logits or routing decisions** (which expert(s) each token goes to)
3. **May have forward hooks** that capture logits or token distributions

**Examples:**

```python
# Mixtral style
model.layers.0.block_sparse_moe.router          # ← Router is the MoE block itself
model.layers.0.block_sparse_moe.gate             # ← Or a separate gate module

# Custom style
model.layers.0.moe.router                        # Simple naming
model.encoder.layers.5.expert_selector.routing   # Nested routing
model.blocks.0.sparse_gate                       # Different convention
```

### Understanding Your Model

**Print the model:**
```python
print(model)
```

**Look for:**
- Modules containing "moe", "gate", "router", "sparse", "expert"
- Modules that wrap multiple expert layers
- Modules with `forward()` that takes `hidden_states` and returns routing decisions

---

## ⚡ Quick Fix: Manual Router Specification

If you just want to monitor this session, specify the router modules explicitly:

```python
from moewatch import WatchConfig, MoEWatch

config = WatchConfig(
    router_modules=[
        "model.layers.0.moe_router",
        "model.layers.1.moe_router",
        "model.layers.2.moe_router",
        # ... add all your router layers
    ]
)

watcher = MoEWatch(model, config=config)
watcher.start()

# Your training code here
```

**Verify it works:**
```python
alerts = watcher.step(0)  # Should not raise
if alerts:
    print("✅ Router detection successful!")
    for alert in alerts:
        print(f"  {alert.message}")
else:
    print("⚠️  No alerts (might still be working)")
```

---

## 🏗️ Permanent Fix: Contribute to Auto-Detection Registry

Make your architecture discoverable by **all MoEWatch users** forever.

### Step 1: Identify Your Router Class Names

Find the **Python class names** of your router modules:

```python
for name, module in model.named_modules():
    if "router" in name.lower():
        class_name = type(module).__name__
        print(f"{name:50} → {class_name}")
```

**Example output:**
```
model.layers.0.moe.router                        → CustomMoEGate
model.layers.1.moe.router                        → CustomMoEGate
model.layers.2.moe.router                        → CustomMoEGate
```

In this case, the router class is `CustomMoEGate`.

### Step 2: Add to the Registry

Edit `moewatch/hooks/detection.py`:

```python
# Find this section (around line 64):
_ARCHITECTURE_REGISTRY: Dict[str, FrozenSet[str]] = {
    "Mixtral": frozenset({"MixtralSparseMoeBlock", ...}),
    "OLMoE": frozenset({"OlmoeMoE", ...}),
    # ... other architectures
}

# Add your architecture:
_ARCHITECTURE_REGISTRY: Dict[str, FrozenSet[str]] = {
    "Mixtral": frozenset({"MixtralSparseMoeBlock", ...}),
    "OLMoE": frozenset({"OlmoeMoE", ...}),
    "YourArchName": frozenset({
        "CustomMoEGate",        # Add your router class names here
        "AlternativeRouterName", # If you have multiple variants
    }),
    # ... other architectures
}
```

### Step 3: Test Auto-Detection

```python
from moewatch.hooks.detection import detect_router_modules

routers = detect_router_modules(model)
print(f"Detected {len(routers)} router modules:")
for router in routers:
    print(f"  ✅ {router}")
```

**Expected output:**
```
Detected 32 router modules:
  ✅ model.layers.0.moe.router
  ✅ model.layers.1.moe.router
  ... (all your routers)
```

### Step 4: Open a Pull Request

Share your contribution with the community!

```bash
git checkout -b add-custom-architecture
git add moewatch/hooks/detection.py
git commit -m "Add support for CustomMoE architecture"
git push origin add-custom-architecture
```

Then [open a PR on GitHub](https://github.com/Abineshabee/moewatch/pulls) with:

```markdown
## Architecture: CustomMoE

Adds auto-detection for CustomMoE models.

**Router class(es):** `CustomMoEGate`, `AlternativeRouterName`

**Models:** Your-Model-8x1B, Your-Model-32x7B

**Tested with:**
- Model: [link to HuggingFace]
- Layers: 32
- Experts per layer: 8
```

---

## 📚 Real-World Examples

### Example 1: Simple Custom MoE

**Architecture:**
```python
class MyModel(nn.Module):
    def __init__(self, num_layers=32, num_experts=8):
        super().__init__()
        self.layers = nn.ModuleList([
            MyMoELayer(num_experts) for _ in range(num_layers)
        ])

class MyMoELayer(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])
        self.gate = MyGatingNetwork(num_experts)  # ← ROUTER CLASS
```

**Class name to register:** `MyGatingNetwork`

**Registry entry:**
```python
"MyModel": frozenset({"MyGatingNetwork"})
```

---

### Example 2: Hierarchical MoE

**Architecture:**
```python
class HierarchicalMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_groups = nn.ModuleList([
            ExpertGroup() for _ in range(4)
        ])
        self.group_router = GroupSelector()          # ← ROUTER 1
        self.expert_routers = nn.ModuleList([
            ExpertRouter() for _ in range(4)         # ← ROUTER 2 (multiple)
        ])
```

**Classes to register:** `GroupSelector`, `ExpertRouter`

**Registry entry:**
```python
"HierarchicalMoE": frozenset({
    "GroupSelector",
    "ExpertRouter",
})
```

---

### Example 3: Top-K with Capacity

**Architecture:**
```python
class TopKMoE(nn.Module):
    def __init__(self, num_experts=16, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])
        self.router = TopKRouter(num_experts, top_k)  # ← ROUTER CLASS
        self.capacity_controller = CapacityFactor()
```

**Class name to register:** `TopKRouter`

**Registry entry:**
```python
"TopKMoE": frozenset({"TopKRouter"})
```

---

## 🧪 Testing Your Addition

### Test 1: Manual Specification (Must Work)

```python
from moewatch import WatchConfig, MoEWatch, audit

model = YourModel()

# Should NOT raise
config = WatchConfig(router_modules=[...])
watcher = MoEWatch(model, config=config)
watcher.start()

# Run a forward pass
batch = {"input_ids": torch.randint(0, 1000, (4, 128))}
with torch.no_grad():
    _ = model(**batch)

# Check health
alerts = watcher.step(0)
print(f"✅ Collected {len(alerts)} alert(s)")

watcher.stop()
```

### Test 2: Auto-Detection (Nice to Have)

```python
from moewatch.hooks.detection import detect_router_modules

routers = detect_router_modules(model)
print(f"Auto-detected {len(routers)} routers")
assert len(routers) > 0, "Auto-detection failed!"
```

### Test 3: Full Audit Flow

```python
from moewatch import audit

report = audit(model, steps=50)
print(report.summary())
```

---

## ❓ Troubleshooting

### Problem: "No routers detected"

**Cause:** Your router class name isn't in the registry, and heuristics didn't catch it.

**Fix 1: Check class name**
```python
for name, module in model.named_modules():
    if "router" in name.lower() or "gate" in name.lower():
        print(f"{name:50} → {type(module).__name__}")
```

**Fix 2: Use manual specification**
```python
config = WatchConfig(
    router_modules=[
        "model.layers.0.your_router_name",
        "model.layers.1.your_router_name",
        # ...
    ]
)
```

**Fix 3: Add to registry (the "right" way)**
```python
# moewatch/hooks/detection.py
_ARCHITECTURE_REGISTRY["YourModel"] = frozenset({"YourRouterClass"})
```

---

### Problem: "Hooks attached but no data collected"

**Cause:** Router module doesn't output routing logits/decisions where hooks expect them.

**Diagnosis:**
```python
import torch

model = YourModel()
watcher = MoEWatch(model, config=WatchConfig(router_modules=[...]))
watcher.start()

batch = {"input_ids": torch.randint(0, 1000, (4, 128))}
with torch.no_grad():
    output = model(**batch)

stats = watcher._collector.get_all_stats()
for layer_name, layer_stats in stats.items():
    print(f"{layer_name}: {layer_stats.event_count} events collected")
```

**If `event_count` is 0:** The hook didn't fire. Possible causes:
- Wrong module name (module doesn't exist)
- Router module is not actually called during forward pass
- Router output doesn't match the expected signature

**Solution:** Check your router's forward signature:
```python
# Router should do something like:
def forward(self, hidden_states):
    logits = self.scoring_network(hidden_states)  # Shape: (batch_size * seq_len, num_experts)
    routing = torch.softmax(logits, dim=-1)
    return routing  # or logits, or both
```

---

### Problem: "Alert levels seem off"

**Cause:** Your model uses different numbers of experts or routing mechanisms.

**Fix:** Customize thresholds for your architecture:

```python
config = WatchConfig(
    # For 64 experts (vs default 8):
    dead_threshold=0.0001,      # Lower threshold
    entropy_warn=0.50,          # Higher entropy (log₂(64) ≈ 6)
    
    # For top-k (vs top-1):
    load_imbalance_error=8.0,   # More lenient (expected imbalance)
)
```

---

## 📋 Checklist for Contributing

- [ ] **Identified router classes:** Class names of your router modules
- [ ] **Tested manual mode:** `WatchConfig(router_modules=[...])` works
- [ ] **Tested auto-detection:** Registry entry + `detect_router_modules()` returns results
- [ ] **Tested full audit:** `audit(model)` completes without errors
- [ ] **Updated registry:** Added your architecture to `_ARCHITECTURE_REGISTRY`
- [ ] **Created PR:** Opened on [GitHub](https://github.com/Abineshabee/moewatch)

---

## 🎓 Understanding the Detection Pipeline

MoEWatch uses a **two-pass strategy** to find routers:

### Pass 1: Architecture Registry (Fast)

```python
# Check if module class name matches a known architecture
for module in model.modules():
    class_name = type(module).__name__
    if class_name in _ALL_KNOWN_CLASSES:
        return [module_path]  # ← Found it!
```

**Pros:** Fast, zero false positives  
**Cons:** Only works for registered architectures

### Pass 2: Heuristic Fallback (Flexible)

```python
# Scan for substring matches: "router", "gate", "moe", etc.
for name, module in model.named_modules():
    class_name = type(module).__name__
    
    # Check for router substrings
    if any(keyword in class_name.lower() for keyword in ["router", "gate", ...]):
        # Filter out obvious non-routers (embedding, norm, attn)
        if not any(exclude in class_name for exclude in EXCLUSIONS):
            return [name]  # ← Likely a router
```

**Pros:** Works for unknown architectures  
**Cons:** May catch false positives

### Why Registry Matters

By adding your architecture to the registry, you **bypass heuristics** and guarantee detection:

```python
# Without registry: heuristic scan (might miss it)
routers = detect_router_modules(model)  # May be empty

# With registry: immediate hit
routers = detect_router_modules(model)  # ✅ Found!
```

---

## 📚 Reference: How Hooks Work

Once your routers are identified, MoEWatch attaches **forward hooks** to capture routing data:

```python
def router_hook(module, input, output):
    """
    Called after the router's forward() completes.
    
    Captures:
    - Logits (raw scores before softmax)
    - Token distribution (which experts got tokens)
    - Batch size, sequence length
    """
    # Extract and aggregate stats
    collector.add_event(
        layer_name="model.layers.0.moe.router",
        logits=output.logits if hasattr(output, 'logits') else output,
        expert_counts=compute_expert_counts(output),
    )
```

Your router just needs to:
1. Accept `hidden_states` (or equivalent) as input
2. Return logits or routing decisions
3. Not modify the model's parameters

---

## 🤝 Contributing Guidelines

When you submit a PR to add a new architecture:

```markdown
## Add Support for [ArchitectureName]

**Router class(es):** 
- `YourRouterClass1`
- `YourRouterClass2`

**Models:** 
- [Link to HuggingFace hub](https://huggingface.co)

**Tested on:**
- Layers: 32
- Experts per layer: 8
- Batch size: 4
- Seq length: 512

**Verification:**
```python
from moewatch import audit
report = audit(model, steps=100)
print(report.summary())
```

## Example Output
[Paste console output here]
```

---

## 🔗 Next Steps

- **Just experimenting?** → Use [manual specification](#quick-fix-manual-router-specification)
- **Using a standard model?** → Check if it's already [supported](../README.md#supported-architectures)
- **Building for production?** → [Contribute your architecture](#permanent-fix-contribute-to-auto-detection-registry)
- **Need help?** → [Open an issue on GitHub](https://github.com/Abineshabee/moewatch/issues)

---

## 📖 Additional Resources

- [Getting Started](./quickstart.md) — First-time users
- [Configuration Reference](./config.md) — Tuning thresholds
- [API Reference](./api_reference.md) — Full method documentation
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Development guidelines

---

**Happy monitoring!** If you add support for a new architecture, we'd love a PR. 🎉

Email: abineshabee2@gmail.com  
GitHub: [Abineshabee/moewatch](https://github.com/Abineshabee/moewatch)
