"""
examples/deepseek_example.py
============================

MoEWatch — DeepSeek-MoE integration example
-------------------------------------------

This file shows how to use MoEWatch with a DeepSeek-MoE model.
Demonstrates all three integration modes:

  1.  Offline audit          — one-shot diagnostic
  2.  Live monitor           — custom training loop
  3.  HuggingFace Trainer    — attach as TrainerCallback

Model: Tiny DeepSeek-MoE built from config (NO download needed)
  - 2 transformer layers, 4 experts per layer
  - Uses DeepseekV2 gate-based routing
  - ~400 K random parameters, runs on CPU in seconds

Usage:
  python examples/deepseek_example.py                # all three demos
  python examples/deepseek_example.py --demo audit   # offline audit only
  python examples/deepseek_example.py --demo live    # live monitor only
  python examples/deepseek_example.py --demo trainer # HF Trainer only
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

import moewatch
from moewatch import MoEWatch, WatchConfig, audit


# ---------------------------------------------------------------------------
# Tiny model config — no download required
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 1_000
SEQ_LEN     = 32
BATCH_SIZE  = 2


def build_tiny_deepseek() -> AutoModelForCausalLM:
    """Build a tiny DeepSeek-MoE model with random weights (~400 K params)."""
    try:
        from transformers import DeepseekV2Config
        
        cfg = DeepseekV2Config(
            hidden_size             = 64,
            intermediate_size       = 128,
            num_hidden_layers       = 2,
            num_attention_heads     = 2,
            num_key_value_heads     = 2,
            moe_intermediate_size   = 128,
            num_experts             = 4,      # 4 experts
            num_experts_per_tok     = 2,      # top-2 routing
            vocab_size              = VOCAB_SIZE,
            max_position_embeddings = 128,
        )
        model = AutoModelForCausalLM.from_config(cfg)
    except (ImportError, ValueError) as e:
        print(f"  ⚠ DeepseekV2Config not available: {e}")
        print("  Falling back to Mixtral for demonstration...")
        from transformers import MixtralConfig, MixtralForCausalLM
        
        cfg = MixtralConfig(
            hidden_size             = 64,
            intermediate_size       = 128,
            num_hidden_layers       = 2,
            num_attention_heads     = 2,
            num_key_value_heads     = 2,
            num_local_experts       = 4,
            num_experts_per_tok     = 2,
            vocab_size              = VOCAB_SIZE,
            max_position_embeddings = 128,
        )
        model = MixtralForCausalLM(cfg)
    
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"  ✓ Tiny DeepSeek-MoE built  ({n_params:.1f} K params, 4 experts × 2 layers)")
    return model


# ---------------------------------------------------------------------------
# Tiny synthetic dataset
# ---------------------------------------------------------------------------

class RandomTokenDataset(Dataset):
    """Generates random token sequences for testing."""
    def __init__(self, n_samples: int = 32, seq_len: int = SEQ_LEN, vocab_size: int = VOCAB_SIZE):
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
# Demo 1 — Offline audit
# ===========================================================================

def demo_offline_audit(model):
    """One-shot offline diagnostic with audit()."""
    print("\n" + "━"*60)
    print("  DEMO 1 — Offline audit  (moewatch.audit)")
    print("━"*60)

    dataset    = RandomTokenDataset(n_samples=20)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n[A] audit() with WatchConfig.default() …\n")
    report = audit(
        model,
        dataloader,
        steps=10,
        config=WatchConfig(sample_every=1),
    )

    print("\n── Text summary ──────────────────────────────────────────")
    print(report.summary())

    print("── Overall health ────────────────────────────────────────")
    print(f"  report.overall_health  = {report.overall_health}")
    print(f"  report.n_layers        = {report.n_layers}")
    print(f"  report.n_experts_total = {report.n_experts_total}")

    print("\n── Dead / cold experts ───────────────────────────────────")
    dead = report.dead_experts(include_cold=True)
    if dead:
        for entry in dead:
            print(f"  {entry}")
    else:
        print("  ✓ All experts active — no collapse detected.")

    print("\n── Per-layer routing entropy ─────────────────────────────")
    for layer_name, result in report.routing_entropy().items():
        print(
            f"  {layer_name[-44:]:<44s}  "
            f"H={result.entropy_bits:.3f} bits  "
            f"({result.entropy_norm*100:.1f}% of max)  "
            f"[{result.alert_level.value}]  trend={result.trend}"
        )

    print("\n── Per-layer utilization ─────────────────────────────────")
    for layer_name, util in report.utilization().items():
        util_pcts = "  ".join(f"E{i}:{v*100:.1f}%" for i, v in enumerate(util.utilization))
        print(
            f"  {layer_name[-44:]:<44s}  "
            f"imbalance={util.load_imbalance_score:.2f}×\n"
            f"    {util_pcts}"
        )

    print("\n── Recommendations ───────────────────────────────────────")
    for rec in report.recommendations():
        print(f"  💡 {rec[:120]}")

    # Test silent mode
    print("\n\n[B] audit() in silent mode …")
    report_silent = audit(
        model,
        dataloader,
        steps=10,
        config=WatchConfig.silent(),
        verbose=False,
    )
    print(f"  ✓ Silent audit done — {report_silent!r}")

    # Test JSON export
    print("\n\n[C] Exporting report to JSON …")
    json_str = report.to_json(indent=2)
    print(f"  JSON length: {len(json_str):,} chars  (first 300 chars shown)")
    print(json_str[:300], "…")

    print("\n  ✓ Demo 1 complete.\n")
    return report


# ===========================================================================
# Demo 2 — Live monitor
# ===========================================================================

def demo_live_monitor(model):
    """Live monitoring in a custom training loop."""
    print("\n" + "━"*60)
    print("  DEMO 2 — Live monitor  (MoEWatch + custom loop)")
    print("━"*60)

    dataset    = RandomTokenDataset(n_samples=32)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    config = WatchConfig(
        sample_every = 1,
        log_every    = 5,
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

    # Context manager syntax
    print("\n\n[B] Context manager syntax …")
    cm_config = WatchConfig(output="silent", sample_every=1, log_every=2)
    with MoEWatch(model, config=cm_config) as w:
        for step, batch in enumerate(DataLoader(dataset, batch_size=BATCH_SIZE)):
            with torch.no_grad():
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            w.step(step)
    print(f"  ✓ Context manager done — {len(w.get_alert_log())} alerts recorded")

    print("\n  ✓ Demo 2 complete.\n")


# ===========================================================================
# Demo 3 — HuggingFace Trainer integration
# ===========================================================================

def demo_hf_trainer(model):
    """Integration with HuggingFace Trainer."""
    print("\n" + "━"*60)
    print("  DEMO 3 — HuggingFace Trainer  (watcher.attach(trainer))")
    print("━"*60)

    dataset = RandomTokenDataset(n_samples=32)

    training_args = TrainingArguments(
        output_dir                  = "/tmp/moewatch_deepseek_demo",
        num_train_epochs            = 1,
        per_device_train_batch_size = BATCH_SIZE,
        logging_steps               = 4,
        save_steps                  = 9999,
        report_to                   = "none",
        use_cpu                     = True,
        dataloader_num_workers      = 0,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = dataset,
    )

    config  = WatchConfig(sample_every=1, log_every=4, output="console")
    watcher = MoEWatch(model, config=config)
    watcher.attach(trainer)

    print("\n  Starting Trainer.train() with MoEWatch attached …\n")
    trainer.train()

    print(f"\n── watcher.summary() ─────────────────────────────────────")
    print(f"  {watcher.summary()}")

    alert_log = watcher.get_alert_log()
    print(f"\n── Last 5 alerts (of {len(alert_log)} total) ─────────────────")
    for alert in alert_log[-5:]:
        print(f"  step={alert.step:<4d}  [{alert.level.value}]  {alert.message[:80]}")

    print("\n  ✓ Demo 3 complete.\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MoEWatch — DeepSeek-MoE integration example"
    )
    parser.add_argument(
        "--demo",
        choices=["audit", "live", "trainer", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    args = parser.parse_args()

    print(f"\n  moewatch version : {moewatch.__version__}")
    print(f"  torch version    : {torch.__version__}")

    model = build_tiny_deepseek()

    if args.demo in ("audit", "all"):
        demo_offline_audit(model)

    if args.demo in ("live", "all"):
        demo_live_monitor(model)

    if args.demo in ("trainer", "all"):
        demo_hf_trainer(model)

    print("\n" + "="*60)
    print("  All demos complete — moewatch is working correctly.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
