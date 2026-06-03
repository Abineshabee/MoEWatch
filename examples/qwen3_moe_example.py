"""
examples/qwen3_moe_example.py
==============================

MoEWatch — Qwen3-MoE integration example
-----------------------------------------

This file exercises all three ways to use MoEWatch with a Qwen3-MoE model:

  1.  Offline audit          — one-shot diagnostic, no training needed
  2.  Live monitor           — custom training loop with watcher.step()
  3.  Collapsed model audit  — synthetic expert collapse, so you can see
                               what CRITICAL / ERROR output looks like

Model used: tiny Qwen3-MoE built entirely from config (NO download needed).
  - 2 transformer layers, 4 experts per layer, top-2 routing.
  - ~749 K random parameters, runs on CPU in seconds.
  - Qwen3MoeTopKRouter gates are auto-detected by moewatch's registry
    (fixed in v0.1.0+: previously pointed at the parent MLP block which
     returns only hidden states, not routing logits).

Architecture note
-----------------
  In Qwen3-MoE (and Qwen2-MoE) each MoE layer is structured as:

    model.layers.{i}.mlp                  ← Qwen3MoeSparseMoeBlock  (hidden → hidden)
    model.layers.{i}.mlp.gate             ← Qwen3MoeTopKRouter       ← hooked here
    model.layers.{i}.mlp.experts          ← Qwen3MoeExperts

  The gate returns a 3-tuple: (logits, routing_weights, expert_indices).
  moewatch's RouterHook._from_tuple_output picks up the float32 logits at
  position [0] and the int64 indices at position [2], computes per-expert
  token counts, and writes a RoutingEvent to the ring buffer.

Usage
-----
  python examples/qwen3_moe_example.py               # all three demos
  python examples/qwen3_moe_example.py --demo audit  # offline audit only
  python examples/qwen3_moe_example.py --demo live   # live monitor only
  python examples/qwen3_moe_example.py --demo collapse  # collapsed model
"""

from __future__ import annotations

import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

import moewatch
from moewatch import MoEWatch, WatchConfig, audit
from moewatch.hooks.detection import detect_router_modules

# Suppress the sample_every=1 perf warning — acceptable for a tiny demo model
warnings.filterwarnings(
    "ignore",
    message=".*sample_every=1.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 1_000
SEQ_LEN    = 32
BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Tiny Qwen3-MoE model — no download required
# ---------------------------------------------------------------------------

def build_tiny_qwen3moe() -> Qwen3MoeForCausalLM:
    """Build a 2-layer, 4-expert Qwen3-MoE model with random weights (~749 K params).

    Key parameters
    --------------
    num_experts         : 4  (experts per MoE layer)
    num_experts_per_tok : 2  (top-2 routing, like the real Qwen3-MoE-*B models)
    decoder_sparse_step : 1  (every layer is a MoE layer; default skips some)
    output_router_logits: True — exposes logits in model output (not needed by
                          moewatch hooks, but useful for debugging)
    """
    cfg = Qwen3MoeConfig(
        hidden_size             = 128,
        intermediate_size       = 256,
        moe_intermediate_size   = 128,
        num_hidden_layers       = 2,
        num_attention_heads     = 4,
        num_key_value_heads     = 2,
        num_experts             = 4,
        num_experts_per_tok     = 2,
        decoder_sparse_step     = 1,    # make every layer an MoE layer
        vocab_size              = VOCAB_SIZE,
        max_position_embeddings = 128,
        output_router_logits    = True,
    )
    model = Qwen3MoeForCausalLM(cfg)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"  ✓ Tiny Qwen3-MoE built  ({n_params:.1f} K params, 4 experts × 2 layers, top-2)")
    return model


# ---------------------------------------------------------------------------
# Tiny synthetic dataset — random token IDs, no real text needed
# ---------------------------------------------------------------------------

class RandomTokenDataset(Dataset):
    """Generates *n_samples* random sequences of *seq_len* token IDs."""

    def __init__(
        self,
        n_samples: int = 32,
        seq_len: int = SEQ_LEN,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        self.data = []
        for _ in range(n_samples):
            ids = torch.randint(0, vocab_size, (seq_len,))
            self.data.append({
                "input_ids":      ids,
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "labels":         ids.clone(),
            })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ===========================================================================
# Demo 0 — Architecture inspection: show what moewatch auto-detects
# ===========================================================================

def demo_architecture_inspection(model: Qwen3MoeForCausalLM) -> None:
    """Show how moewatch discovers the Qwen3MoeTopKRouter gate modules."""

    print("\n" + "━" * 60)
    print("  DEMO 0 — Architecture inspection")
    print("━" * 60)

    print("\n  All MoE-related modules in the tiny Qwen3-MoE:")
    for name, module in model.named_modules():
        cls = type(module).__name__
        if any(kw in cls for kw in ("Moe", "MoE", "Router", "Expert", "Gate", "Sparse")):
            print(f"    {name:<50s}  {cls}")

    print("\n  Modules auto-detected by moewatch (registry pass):")
    detected = detect_router_modules(model)
    for name in detected:
        cls = type(dict(model.named_modules())[name]).__name__
        print(f"    {name:<50s}  {cls}")

    print(
        "\n  Note: moewatch hooks the Qwen3MoeTopKRouter (gate), NOT the parent\n"
        "  Qwen3MoeSparseMoeBlock (mlp). The gate returns\n"
        "  (logits, routing_weights, expert_indices) so routing stats are\n"
        "  captured correctly. The mlp block returns only hidden states.\n"
    )


# ===========================================================================
# Demo 1 — Offline audit
# ===========================================================================

def demo_offline_audit(model: Qwen3MoeForCausalLM) -> None:
    """
    moewatch.audit() is the simplest entry point.

    Pass the model (and optionally a DataLoader); it attaches hooks,
    runs N forward passes, then returns a structured AuditReport.
    No training loop required.
    """
    print("\n" + "━" * 60)
    print("  DEMO 1 — Offline audit  (moewatch.audit)")
    print("━" * 60)

    dataset    = RandomTokenDataset(n_samples=20)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── A: default config ─────────────────────────────────────────────────────
    print("\n[A] audit() with default config …\n")
    report = audit(
        model,
        dataloader,
        steps=10,
        config=WatchConfig(sample_every=1),   # capture every pass in this tiny demo
    )

    print("\n── Text summary ──────────────────────────────────────────")
    print(report.summary())

    print("\n── Overall health ────────────────────────────────────────")
    print(f"  report.overall_health  = {report.overall_health}")
    print(f"  report.n_layers        = {report.n_layers}")
    print(f"  report.n_experts_total = {report.n_experts_total}")

    print("\n── Dead / cold experts ───────────────────────────────────")
    dead = report.dead_experts(include_cold=True)
    if dead:
        for entry in dead:
            print(f"  Layer {entry.layer_idx}  Expert {entry.expert_idx}  "
                  f"state={entry.state.value}  util={entry.utilization_pct:.2f}%")
    else:
        print("  ✓ All experts active — no collapse detected.")

    print("\n── Per-layer routing entropy ─────────────────────────────")
    for name, result in report.routing_entropy().items():
        print(
            f"  {name[-46:]:<46s}  "
            f"H={result.entropy_bits:.3f}b  "
            f"({result.entropy_norm * 100:.1f}% of max)  "
            f"[{result.alert_level.value}]  trend={result.trend}"
        )

    print("\n── Per-layer utilization ─────────────────────────────────")
    for name, util in report.utilization().items():
        util_pcts = "  ".join(
            f"E{i}:{v * 100:.1f}%" for i, v in enumerate(util.utilization)
        )
        print(
            f"  {name[-46:]:<46s}  "
            f"imbalance={util.load_imbalance_score:.2f}×\n"
            f"    {util_pcts}"
        )

    print("\n── Recommendations ───────────────────────────────────────")
    recs = report.recommendations()
    if recs:
        for rec in recs:
            print(f"  💡 {rec[:120]}")
    else:
        print("  ✓ No recommendations — routing looks healthy.")

    # ── B: aggressive config ──────────────────────────────────────────────────
    print("\n\n[B] audit() with WatchConfig.aggressive() …\n")
    report_agg = audit(
        model,
        dataloader,
        steps=10,
        config=WatchConfig.aggressive(),
    )
    print(
        f"  overall_health = {report_agg.overall_health}\n"
        f"  dead={report_agg.n_dead}  cold={report_agg.n_cold}  "
        f"healthy={report_agg.n_healthy}"
    )

    # ── C: silent mode ────────────────────────────────────────────────────────
    print("\n\n[C] audit() in silent mode …")
    report_silent = audit(
        model,
        dataloader,
        steps=10,
        config=WatchConfig.silent(),
        verbose=False,
    )
    print(f"  ✓ Silent audit done — {report_silent!r}")

    # ── D: no dataloader (synthetic batches) ─────────────────────────────────
    print("\n\n[D] audit() without a dataloader (synthetic random input) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report_synth = audit(model, steps=10, verbose=False)
    print(f"  ✓ Synthetic audit done — health: {report_synth.overall_health}")

    # ── E: JSON export ─────────────────────────────────────────────────────────
    print("\n\n[E] JSON export …")
    json_str = report.to_json(indent=2)
    print(f"  JSON length: {len(json_str):,} chars  (first 300 chars shown)")
    print(f"  {json_str[:300]} …")

    print("\n  ✓ Demo 1 complete.\n")


# ===========================================================================
# Demo 2 — Live monitor in a custom training loop
# ===========================================================================

def demo_live_monitor(model: Qwen3MoeForCausalLM) -> None:
    """
    MoEWatch.start() / step() / stop() — works with any custom training loop.

    step() returns a List[Alert]: empty when healthy, populated otherwise.
    You can react programmatically: log to W&B, raise, save checkpoint, etc.
    """
    print("\n" + "━" * 60)
    print("  DEMO 2 — Live monitor  (MoEWatch + custom loop)")
    print("━" * 60)

    dataset    = RandomTokenDataset(n_samples=32)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    config = WatchConfig(
        sample_every = 1,   # instrument every forward pass
        log_every    = 5,   # emit diagnostics every 5 steps
        output       = "console",
    )

    watcher = MoEWatch(model, config=config)
    watcher.start()

    print(f"\n  Running {len(dataloader)} training steps …\n")

    all_alerts = []
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(
            input_ids      = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            labels         = batch["labels"],
        )
        outputs.loss.backward()
        optimizer.step()

        alerts = watcher.step(step)
        all_alerts.extend(alerts)

    watcher.stop()
    model.eval()

    # ── Summary ───────────────────────────────────────────────────────────────
    errors = [a for a in all_alerts if a.level.value == "ERROR"]
    warns  = [a for a in all_alerts if a.level.value == "WARN"]
    infos  = [a for a in all_alerts if a.level.value == "INFO"]

    print(f"\n── Alert summary ─────────────────────────────────────────")
    print(f"  total={len(all_alerts)}  ERROR={len(errors)}  WARN={len(warns)}  INFO={len(infos)}")

    print("\n── Last 3 alerts ─────────────────────────────────────────")
    for alert in watcher.get_alert_log()[-3:]:
        print(f"  step={alert.step:<4d}  [{alert.level.value}]  {alert.message[:80]}")

    print(f"\n── watcher.summary() ─────────────────────────────────────")
    print(f"  {watcher.summary()}")

    # ── Context manager syntax ─────────────────────────────────────────────────
    print("\n\n[B] Context manager syntax …")
    cm_config = WatchConfig(output="silent", sample_every=1, log_every=2)
    with MoEWatch(model, config=cm_config) as w:
        for step, batch in enumerate(DataLoader(dataset, batch_size=BATCH_SIZE)):
            with torch.no_grad():
                model(
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                )
            w.step(step)
    print(f"  ✓ Context manager done — {len(w.get_alert_log())} alerts recorded")

    # ── JSON alert log ──────────────────────────────────────────────────────────
    print("\n[C] Alert log as JSON (first 200 chars) …")
    log_json = watcher.get_alert_log_json()
    print(f"  {log_json[:200]} …")

    print("\n  ✓ Demo 2 complete.\n")


# ===========================================================================
# Demo 3 — Collapsed model: what ERROR output looks like
# ===========================================================================

def demo_collapsed_model() -> None:
    """
    Build a synthetic MoE model where one expert always gets 100% of traffic
    and run audit() to show what CRITICAL health and ERROR alerts look like.

    This demo does NOT use the real Qwen3-MoE architecture — it uses the
    same SyntheticMoERouter from the test suite so it runs in milliseconds.
    """
    print("\n" + "━" * 60)
    print("  DEMO 3 — Collapsed model (what ERROR output looks like)")
    print("━" * 60)

    # ---------------------------------------------------------------------------
    # Inline synthetic model — same design as tests/conftest.py
    # ---------------------------------------------------------------------------

    class _Expert(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.fc = nn.Linear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    class _CollapsedRouter(nn.Module):
        """Router that forces 100% of traffic to expert 0 — simulates collapse."""

        def __init__(self, hidden_dim: int = 64, n_experts: int = 8) -> None:
            super().__init__()
            self.n_experts = n_experts
            self.gate = nn.Linear(hidden_dim, n_experts, bias=False)
            self.experts = nn.ModuleList([_Expert(hidden_dim) for _ in range(n_experts)])

        def forward(self, x: torch.Tensor) -> tuple:
            B, S, D = x.shape
            flat = x.reshape(B * S, D)
            logits = self.gate(flat)

            # Force collapse: expert 0 always wins
            with torch.no_grad():
                logits.fill_(-1e9)
                logits[:, 0] = 1e9

            # Route through expert 0 only
            out = self.experts[0](flat).reshape(B, S, D)
            return out, logits   # tuple — RouterHook Strategy 1

    class _CollapsedMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(256, 64)
            self.routers = nn.ModuleList([_CollapsedRouter() for _ in range(2)])
            self.lm_head = nn.Linear(64, 256, bias=False)

        def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
            x = self.embedding(input_ids)
            for router in self.routers:
                x, _ = router(x)
            return self.lm_head(x)

    collapsed_model = _CollapsedMoE()
    collapsed_model.eval()

    router_names = ["routers.0", "routers.1"]

    print("\n  Running audit on a model where expert 0 receives 100% of traffic …\n")

    batches = [
        {
            "input_ids":      torch.randint(0, 256, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        }
        for _ in range(20)
    ]

    config = WatchConfig(
        router_modules = router_names,
        sample_every   = 1,
        output         = "console",
    )

    report = audit(collapsed_model, batches, steps=15, config=config)

    print("\n── AuditReport query results ─────────────────────────────")
    print(f"  overall_health = {report.overall_health}")
    print(f"  has_collapse   = {report.has_collapse}")
    print(f"  n_dead         = {report.n_dead}")
    print(f"  n_cold         = {report.n_cold}")

    print("\n── Dead expert entries ───────────────────────────────────")
    for entry in report.dead_experts(include_cold=False):
        print(
            f"  Layer {entry.layer_idx}  Expert {entry.expert_idx}  "
            f"util={entry.utilization_pct:.3f}%  [{entry.state.value}]"
        )

    print("\n── Recommendations ───────────────────────────────────────")
    for rec in report.recommendations():
        print(f"  💡 {rec[:120]}")

    print("\n  ✓ Demo 3 complete.\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoEWatch — Qwen3-MoE integration example"
    )
    parser.add_argument(
        "--demo",
        choices=["inspect", "audit", "live", "collapse", "all"],
        default="all",
        help=(
            "Which demo to run  "
            "(inspect | audit | live | collapse | all, default: all)"
        ),
    )
    args = parser.parse_args()

    print(f"\n  moewatch version  : {moewatch.__version__}")
    print(f"  torch version     : {torch.__version__}")

    # Build the shared model (except the collapse demo which uses its own)
    model = build_tiny_qwen3moe()

    if args.demo in ("inspect", "all"):
        demo_architecture_inspection(model)

    if args.demo in ("audit", "all"):
        demo_offline_audit(model)

    if args.demo in ("live", "all"):
        demo_live_monitor(model)

    if args.demo in ("collapse", "all"):
        demo_collapsed_model()

    print("\n" + "=" * 60)
    print("  All demos complete — moewatch is working correctly.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
