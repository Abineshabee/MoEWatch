"""
examples/transformers_example.py
=================================

MoEWatch — HuggingFace Transformers integration example
--------------------------------------------------------

This file shows all three ways to use MoEWatch with a real MoE model:

  1.  Offline audit          — one-shot diagnostic, no training needed
  2.  Live monitor           — custom training loop with watcher.step()
  3.  HuggingFace Trainer    — attach as a TrainerCallback in one line

Model used: tiny Mixtral built entirely from config (NO download needed).
  - 2 transformer layers, 4 experts per layer, 2 experts per token (top-2).
  - ~358 K random parameters, runs on CPU in seconds.
  - Matches the real Mixtral architecture so moewatch auto-detects routers.

Usage:
  python examples/transformers_example.py                # all three demos
  python examples/transformers_example.py --demo audit   # offline audit only
  python examples/transformers_example.py --demo live    # live monitor only
  python examples/transformers_example.py --demo trainer # HF Trainer only
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    MixtralConfig,
    MixtralForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

import moewatch
from moewatch import MoEWatch, WatchConfig, audit


# ---------------------------------------------------------------------------
# Tiny model config — no download required
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 1_000   # small vocab for a toy tokeniser
SEQ_LEN     = 32
BATCH_SIZE  = 2

def build_tiny_mixtral() -> MixtralForCausalLM:
    """Build a 2-layer, 4-expert Mixtral model with random weights (~358 K params)."""
    cfg = MixtralConfig(
        hidden_size             = 64,
        intermediate_size       = 128,
        num_hidden_layers       = 2,
        num_attention_heads     = 2,
        num_key_value_heads     = 2,
        num_local_experts       = 4,    # 4 experts per MoE layer
        num_experts_per_tok     = 2,    # top-2 routing
        vocab_size              = VOCAB_SIZE,
        max_position_embeddings = 128,
    )
    model = MixtralForCausalLM(cfg)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"  ✓ Tiny Mixtral built  ({n_params:.1f} K params, 4 experts × 2 layers)")
    return model


# ---------------------------------------------------------------------------
# Tiny synthetic dataset — random token IDs, no real text needed
# ---------------------------------------------------------------------------

class RandomTokenDataset(Dataset):
    """
    Generates *n_samples* random sequences of *seq_len* token IDs.
    Each sample includes input_ids, attention_mask, and labels (= input_ids).
    """
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

def demo_offline_audit(model: MixtralForCausalLM):
    """
    moewatch.audit() is the simplest entry point.

    Pass the model (and optionally a DataLoader); it attaches hooks,
    runs N forward passes, then returns a structured AuditReport.
    No training loop, no Trainer, no dataset required.
    """
    print("\n" + "━"*60)
    print("  DEMO 1 — Offline audit  (moewatch.audit)")
    print("━"*60)

    dataset    = RandomTokenDataset(n_samples=16)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── A: default config ────────────────────────────────────────────────────
    print("\n[A] audit() with WatchConfig.default() …\n")
    report = audit(
        model,
        dataloader,
        steps=10,
        # sample_every=1 so every forward pass is instrumented.
        # (Default is 10, which would miss all 8 steps in this tiny demo.)
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

    # ── B: aggressive config (tighter thresholds, every-step sampling) ───────
    print("\n\n[B] audit() with WatchConfig.aggressive() …\n")
    report_agg = audit(
        model,
        dataloader,
        steps=10,
        config=WatchConfig.aggressive(),
    )
    print(f"  overall_health (aggressive) = {report_agg.overall_health}")
    print(f"  dead={report_agg.n_dead}  cold={report_agg.n_cold}  healthy={report_agg.n_healthy}")

    # ── C: silent mode (no real-time output, API use) ────────────────────────
    print("\n\n[C] audit() in silent mode …")
    report_silent = audit(
        model,
        dataloader,
        steps=10,
        config=WatchConfig.silent(),
        verbose=False,
    )
    print(f"  ✓ Silent audit done — {report_silent!r}")

    # ── D: audit() without a dataloader (synthetic random batches) ───────────
    print("\n\n[D] audit() with no dataloader (synthetic input) …")
    report_synth = audit(model, steps=5, verbose=False)
    print(f"  ✓ Synthetic audit done — {report_synth!r}")

    # ── E: JSON export ────────────────────────────────────────────────────────
    print("\n\n[E] Exporting report to JSON …")
    json_str = report.to_json(indent=2)
    print(f"  JSON length: {len(json_str):,} chars  (first 300 chars shown)")
    print(json_str[:300], "…")

    print("\n  ✓ Demo 1 complete.\n")
    return report


# ===========================================================================
# Demo 2 — Live monitor in a custom training loop
# ===========================================================================

def demo_live_monitor(model: MixtralForCausalLM):
    """
    MoEWatch sits beside any custom training loop.

    Call watcher.start() before the loop, watcher.step(step) every iteration,
    and watcher.stop() when done.  step() returns a list[Alert] — empty when
    healthy — so you can react programmatically (raise, log to W&B, etc.).
    """
    print("\n" + "━"*60)
    print("  DEMO 2 — Live monitor  (MoEWatch + custom loop)")
    print("━"*60)

    dataset    = RandomTokenDataset(n_samples=32)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    config = WatchConfig(
        sample_every = 1,    # instrument every forward pass (tiny model — fast)
        log_every    = 5,    # emit summary every 5 steps
        output       = "console",
    )

    # ── Start monitoring ─────────────────────────────────────────────────────
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

        # ── Ask MoEWatch to check health ─────────────────────────────────
        alerts = watcher.step(step)
        all_alerts.extend(alerts)

    watcher.stop()
    model.eval()

    # ── Post-run inspection ──────────────────────────────────────────────────
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

    # ── Context manager syntax (alternative) ─────────────────────────────────
    print("\n\n[B] Context manager syntax …")
    cm_config = WatchConfig(output="silent", sample_every=1, log_every=2)
    with MoEWatch(model, config=cm_config) as w:
        for step, batch in enumerate(DataLoader(dataset, batch_size=BATCH_SIZE)):
            with torch.no_grad():
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            w.step(step)
    print(f"  ✓ Context manager done — {len(w.get_alert_log())} alerts recorded")

    # ── Accessing the alert log as JSON ──────────────────────────────────────
    print("\n[C] Alert log as JSON (first 200 chars) …")
    log_json = watcher.get_alert_log_json()
    print(f"  {log_json[:200]} …")

    print("\n  ✓ Demo 2 complete.\n")


# ===========================================================================
# Demo 3 — HuggingFace Trainer integration
# ===========================================================================

def demo_hf_trainer(model: MixtralForCausalLM):
    """
    For HuggingFace Trainer users, one extra line is all you need:

        watcher = MoEWatch(model)
        watcher.attach(trainer)

    MoEWatch injects a MoEWatchCallback.  It calls watcher.step() at every
    Trainer logging step.  Hooks are removed automatically on train end.
    """
    print("\n" + "━"*60)
    print("  DEMO 3 — HuggingFace Trainer  (watcher.attach(trainer))")
    print("━"*60)

    dataset = RandomTokenDataset(n_samples=32)

    training_args = TrainingArguments(
        output_dir                  = "/tmp/moewatch_trainer_demo",
        num_train_epochs            = 1,
        per_device_train_batch_size = BATCH_SIZE,
        logging_steps               = 4,      # MoEWatch ticks on each log step
        save_steps                  = 9999,   # disable checkpointing for demo
        report_to                   = "none",
        use_cpu                     = True,
        dataloader_num_workers      = 0,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = dataset,
    )

    # ── One-line integration ──────────────────────────────────────────────────
    config  = WatchConfig(sample_every=1, log_every=4, output="console")
    watcher = MoEWatch(model, config=config)
    watcher.attach(trainer)    # <── that's it

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
        description="MoEWatch — HuggingFace Transformers integration example"
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

    model = build_tiny_mixtral()

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
