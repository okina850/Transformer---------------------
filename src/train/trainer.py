r"""Phase 1 の学習ループ本体。

Pre-registration（``docs/experiment_log.md``）に定めた手続きをそのまま実装する:

- 毎ステップ: training loss、**クリップ前**の勾配ノルムを記録
- 評価タイミングごと: validation loss、pre-activation 分位点（RQ5）、
  `A(x)` 符号変化数（RQ3、rational 条件のみ）
- 発散を検出したら**そのステップで打ち切り**、発散として記録する
  （個別条件の学習率変更による救済はしない — 呼び出し側の責務）
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from src.train.data import TokenDataset
from src.train.diagnostics import DivergenceMonitor
from src.train.optim import LRSchedule, apply_lr, make_optimizer
from src.train.preact_probe import collect_preact_quantiles


def train_one_run(
    model: nn.Module,
    cfg: dict,
    run_dir: Path,
    device: torch.device,
    seed: int,
) -> dict:
    """1条件・1 seed 分の学習を実行し、ログを ``run_dir`` に書き出す。"""
    tr = cfg["train"]
    data_dir = Path(cfg["data_dir"])
    block_size = model.cfg.block_size

    train_ds = TokenDataset(data_dir / "train.bin", block_size)
    val_ds = TokenDataset(data_dir / "val.bin", block_size)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

    gen = torch.Generator().manual_seed(seed)
    optimizer = make_optimizer(model, lr_theta=tr["lr_theta"], weight_decay=tr.get("weight_decay", 0.0))

    total_steps = int(tr["total_steps"])
    warmup_steps = max(int(total_steps * tr.get("warmup_ratio", 0.01)), 1)
    schedules = {
        "theta": LRSchedule(warmup_steps, total_steps, "linear_decay"),
        "rational": LRSchedule(warmup_steps, total_steps, "constant"),
    }
    base_lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}

    monitor = DivergenceMonitor(vocab_size=meta["vocab_size"])
    clip_norm = float(tr.get("grad_clip_norm", 1.0))

    log_path = run_dir / "step_log.jsonl"
    eval_path = run_dir / "eval_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")
    eval_f = eval_path.open("w", encoding="utf-8")

    eval_interval = int(tr.get("eval_interval", 500))
    eval_batches = int(tr.get("eval_batches", 20))
    batch_size = int(tr["batch_size"])

    model.to(device).train()
    t0 = time.perf_counter()
    stop_reason = "completed"
    final_step = total_steps

    for step in range(total_steps):
        x, y = train_ds.get_batch(batch_size, device, gen)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # クリップ前の生ノルムを記録してからクリップする
        grad_norm_pre_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_norm
        ).item()

        applied_lr = apply_lr(optimizer, step, schedules, base_lrs)
        optimizer.step()

        loss_val = loss.item()
        det = monitor.update(step, loss_val, grad_norm_pre_clip)

        log_f.write(json.dumps({
            "step": step, "loss": loss_val, "grad_norm_pre_clip": grad_norm_pre_clip,
            "lr_theta": applied_lr["theta"], "lr_rational": applied_lr.get("rational"),
            "is_spike": det["is_spike"],
        }) + "\n")

        if det["is_diverged"]:
            stop_reason = f"diverged:{det['reason']}"
            final_step = step
            print(f"  [DIVERGED] step={step} reason={det['reason']} "
                  f"loss={loss_val:.4f} grad_norm={grad_norm_pre_clip:.4f}", flush=True)
            break

        if step % eval_interval == 0 or step == total_steps - 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(eval_batches):
                    vx, vy = val_ds.get_batch(batch_size, device, gen)
                    _, vloss = model(vx, vy)
                    val_losses.append(vloss.item())
                val_loss = sum(val_losses) / len(val_losses)
                val_ppl = math.exp(min(val_loss, 20.0))

                preact = collect_preact_quantiles(model, x)
                sign_changes = [ra.count_sign_changes() for ra in model.rational_activations()]

            model.train()
            elapsed = time.perf_counter() - t0
            eval_f.write(json.dumps({
                "step": step, "val_loss": val_loss, "val_ppl": val_ppl,
                "preact_quantiles": preact, "sign_changes": sign_changes,
                "elapsed_sec": elapsed,
                "tokens_per_sec": (step + 1) * batch_size * block_size / max(elapsed, 1e-9),
            }) + "\n")
            print(f"  step={step:5d}/{total_steps}  train_loss={loss_val:.4f}  "
                  f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.3f}  "
                  f"grad_norm={grad_norm_pre_clip:.3f}  "
                  f"sign_changes={sign_changes[:3]}  ({elapsed:.1f}s)", flush=True)

    log_f.close()
    eval_f.close()

    summary = {
        "stop_reason": stop_reason,
        "final_step": final_step,
        "total_steps_planned": total_steps,
        "elapsed_sec": time.perf_counter() - t0,
        "seed": seed,
        **monitor.summary(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
