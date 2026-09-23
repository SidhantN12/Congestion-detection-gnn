"""Phase 7 overfit test: train CongestionGNN on a single cached graph and
confirm it can memorise it (Spearman correlation > 0.95 on that sample).

Loss: Huber (smooth L1) regression + a pairwise ranking loss, combined as
L_reg + lambda * L_rank. The ranking loss operates on all O(n^2) gcell
pairs within the sample (n <= 289 here, trivial on CPU) - a hinge loss
that penalises the model whenever its predicted order between two gcells
disagrees with their true order, independent of the (harder to fit)
absolute magnitude a pure regression loss chases.

Usage:
    python train.py <graph.pt> [--epochs N] [--lambda-rank F]

Phase 13 added a config-driven trainer for benchmark-scale runs (design-level
splits, checkpoint/resume, gradient accumulation, guarded mixed precision,
early stopping on val Spearman, JSONL metrics log) - see the second half of
this file and notes/phase13-training.md:
    python src/train.py --config configs/synthetic_smoke.toml [--set train.epochs=3] [--resume auto]
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from model import CongestionGNN
from metrics import spearman_corr


def pairwise_ranking_loss(pred, target, margin=0.0):
    """Hinge ranking loss over all pairs, per output channel.

    For every pair (i, j) where target_i != target_j, penalise the model
    if (pred_i - pred_j) doesn't have the same sign as (target_i - target_j),
    by at least `margin`.
    """
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)      # (n, n, C)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)  # (n, n, C)
    target_sign = torch.sign(target_diff)

    loss_per_pair = F.relu(margin - target_sign * pred_diff)
    mask = target_sign != 0
    if mask.sum() == 0:
        return pred.new_zeros(())
    return loss_per_pair[mask].mean()


def combined_loss(pred, target, lambda_rank):
    l_reg = F.smooth_l1_loss(pred, target)
    l_rank = pairwise_ranking_loss(pred, target)
    return l_reg + lambda_rank * l_rank, l_reg.item(), l_rank.item()


def overfit_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-rank", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-spearman", type=float, default=0.95)
    parser.add_argument("--out-prefix", default="data/raw/phase7_overfit")
    parser.add_argument("--save-checkpoint", default=None,
                         help="Path to save the trained model checkpoint (state_dict "
                              "+ architecture config), e.g. for the Phase 8 viewer.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = torch.load(args.graph_path, weights_only=False)
    target = data["gcell"].y_ratio

    in_dims = {nt: data[nt].x.shape[1] for nt in data.node_types}
    edge_types = data.edge_types

    model = CongestionGNN(in_dims, edge_types, hidden_dim=args.hidden_dim,
                           num_layers=args.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"epoch": [], "loss": [], "l_reg": [], "l_rank": [],
               "spearman_h": [], "spearman_v": []}

    print(f"Training on {args.graph_path}: {data['gcell'].x.shape[0]} gcells, "
          f"{sum(p.numel() for p in model.parameters())} model parameters")

    achieved_epoch = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(data)
        loss, l_reg, l_rank = combined_loss(pred, target, args.lambda_rank)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                pred_eval = model(data).numpy()
            target_np = target.numpy()
            sh = spearman_corr(pred_eval[:, 0], target_np[:, 0])
            sv = spearman_corr(pred_eval[:, 1], target_np[:, 1])

            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["l_reg"].append(l_reg)
            history["l_rank"].append(l_rank)
            history["spearman_h"].append(sh)
            history["spearman_v"].append(sv)

            if epoch % 100 == 0 or epoch == args.epochs - 1:
                print(f"epoch {epoch:5d}  loss={loss.item():.4f}  "
                      f"l_reg={l_reg:.4f}  l_rank={l_rank:.4f}  "
                      f"spearman_H={sh:.4f}  spearman_V={sv:.4f}")

            if min(sh, sv) >= args.target_spearman and achieved_epoch is None:
                achieved_epoch = epoch

        if achieved_epoch is not None and epoch >= achieved_epoch + 50:
            # Ran a bit past the threshold to confirm it's stable, then stop.
            break

    model.eval()
    with torch.no_grad():
        pred_final = model(data).numpy()
    target_np = target.numpy()
    final_sh = spearman_corr(pred_final[:, 0], target_np[:, 0])
    final_sv = spearman_corr(pred_final[:, 1], target_np[:, 1])

    print(f"\nFinal: spearman_H={final_sh:.4f}  spearman_V={final_sv:.4f}  "
          f"min={min(final_sh, final_sv):.4f}  target={args.target_spearman}")

    if min(final_sh, final_sv) < args.target_spearman:
        print(f"\nFAIL: could not reach Spearman {args.target_spearman} on this "
              f"single sample - per the gate, this means something is broken "
              f"upstream. STOP and debug rather than continuing to Phase 8.")
        sys.exit(1)

    print(f"\nPASS: model memorised the sample (Spearman >= {args.target_spearman} "
          f"on both channels)")

    if args.save_checkpoint:
        torch.save({
            "state_dict": model.state_dict(),
            "in_dims": in_dims,
            "edge_types": edge_types,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "trained_on": args.graph_path,
        }, args.save_checkpoint)
        print(f"Saved checkpoint to {args.save_checkpoint}")

    # ---- plots ----
    grid_shape = tuple(data["gcell"].grid_shape.numpy())

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].plot(history["epoch"], history["loss"], label="total loss")
    axes[0, 0].plot(history["epoch"], history["l_reg"], label="L_reg (Huber)", alpha=0.7)
    axes[0, 0].plot(history["epoch"], history["l_rank"], label="L_rank (pairwise)", alpha=0.7)
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].set_title("Training loss")
    axes[0, 0].legend()
    axes[0, 0].set_yscale("log")

    axes[0, 1].plot(history["epoch"], history["spearman_h"], label="horizontal")
    axes[0, 1].plot(history["epoch"], history["spearman_v"], label="vertical")
    axes[0, 1].axhline(args.target_spearman, color="red", linestyle="--", alpha=0.5, label="target")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("Spearman correlation")
    axes[0, 1].set_title("Spearman correlation (pred vs actual)")
    axes[0, 1].legend()

    axes[0, 2].axis("off")

    combined_pred = np.maximum(pred_final[:, 0], pred_final[:, 1]).reshape(grid_shape)
    combined_target = np.maximum(target_np[:, 0], target_np[:, 1]).reshape(grid_shape)
    vmax = max(combined_pred.max(), combined_target.max())

    im1 = axes[1, 0].imshow(combined_target, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    axes[1, 0].set_title("Ground truth congestion")
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(combined_pred, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    axes[1, 1].set_title("Predicted congestion")
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    diff = combined_pred - combined_target
    im3 = axes[1, 2].imshow(diff, origin="lower", cmap="RdBu_r",
                             vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[1, 2].set_title(f"Predicted - actual\n(max abs diff={np.abs(diff).max():.4f})")
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    out_png = f"{args.out_prefix}.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


# ---------------------------------------------------------------------------
# Config-driven training (Phase 13): design-level splits, checkpoint/resume,
# gradient accumulation, guarded mixed precision, early stopping on val
# Spearman, JSONL metrics log. See notes/phase13-training.md.
# ---------------------------------------------------------------------------

import contextlib
import copy
import json
import os
import random
import time


def sampled_ranking_loss(pred, target, n_pairs, generator, margin=0.0):
    """pairwise_ranking_loss on n_pairs uniformly sampled gcell pairs instead
    of all n^2 (which Phase 10 measured at 90.3 B/gcell^2 - out of memory
    beyond ~12K gcells). Same hinge; ties (incl. i == j) are excluded, so
    this is an unbiased estimate of the full loss's mean over untied pairs."""
    n = pred.shape[0]
    if n < 2 or n_pairs == 0:
        return pred.new_zeros(())
    i = torch.randint(n, (n_pairs,), generator=generator)
    j = torch.randint(n, (n_pairs,), generator=generator)
    sign = torch.sign(target[i] - target[j])
    loss = F.relu(margin - sign * (pred[i] - pred[j]))
    mask = sign != 0
    return loss[mask].mean() if mask.any() else pred.new_zeros(())


def resolve_device(cfg):
    if cfg.train.device == "cuda" or (cfg.train.device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("train.device='cuda' but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp(cfg, device, model, probe, log):
    """Decide mixed precision. Returns (dtype or None, use_grad_scaler, reason).

    CUDA: "auto"/"fp16" -> float16 + GradScaler, "bf16" -> bfloat16.
    CPU:  "auto" -> fp32 (CPU autocast is rarely a speed-up and not
          supported for every op); "bf16" is tried on a probe forward +
          backward and kept only if it runs and stays finite; "fp16" is
          not used on CPU. Any failure falls back to fp32 with the reason
          logged - never a crash."""
    want = cfg.train.amp
    if want == "off":
        return None, False, "amp=off"
    if device.type == "cuda":
        if want in ("auto", "fp16"):
            return torch.float16, True, f"amp={want} on cuda -> float16 + GradScaler"
        return torch.bfloat16, False, "amp=bf16 on cuda"
    if want == "auto":
        return None, False, "amp=auto on cpu -> fp32"
    if want == "fp16":
        return None, False, "amp=fp16 requested on cpu -> fp32 (float16 autocast not used on CPU)"
    try:
        m = copy.deepcopy(model)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = m(probe)
        out.float().square().mean().backward()
        grads_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        if not (torch.isfinite(out.float()).all() and grads_ok):
            return None, False, "amp=bf16 on cpu: probe produced non-finite values -> fp32"
        return torch.bfloat16, False, "amp=bf16 on cpu: probe forward+backward OK -> bfloat16 autocast"
    except Exception as e:  # graceful degradation is the point
        return None, False, f"amp=bf16 on cpu: probe failed ({type(e).__name__}: {e}) -> fp32"


def autocast_ctx(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else contextlib.nullcontext()


def to_device(g, device):
    return g if device.type == "cpu" else g.to(device)


@torch.no_grad()
def evaluate(model, samples, cfg, device, amp_dtype):
    """Per-sample full-design metrics: units' core predictions are stitched
    back into the whole gcell grid, then Spearman per channel is computed
    over every gcell of the design (metrics.spearman_corr)."""
    from dataset import load_unit, unit_paths
    model.eval()
    rows = []
    for s in samples:
        preds, ys, ids = [], [], []
        for path in unit_paths(s, cfg):
            g = to_device(load_unit(path, cfg.partition.enabled), device)
            with autocast_ctx(device, amp_dtype):
                out = model(g)
            core = g["gcell"].core
            preds.append(out[core].float().cpu())
            ys.append(g["gcell"].y_ratio[core].cpu())
            ids.append(g["gcell"].n_id[core].cpu())
        ids = torch.cat(ids)
        pred = torch.empty(len(ids), 2); y = torch.empty(len(ids), 2)
        pred[ids] = torch.cat(preds); y[ids] = torch.cat(ys)
        sp = [spearman_corr(pred[:, c].numpy(), y[:, c].numpy()) for c in range(2)]
        rows.append(dict(design=s["design"], variant=s["variant"], n_gcells=len(ids),
                         spearman_h=sp[0], spearman_v=sp[1], spearman=(sp[0] + sp[1]) / 2,
                         huber=float(F.smooth_l1_loss(pred, y))))
    by_design = {}
    for r in rows:
        by_design.setdefault(r["design"], []).append(r["spearman"])
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return dict(spearman=mean("spearman"), spearman_h=mean("spearman_h"), spearman_v=mean("spearman_v"),
                huber=mean("huber"), per_design={d: float(np.mean(v)) for d, v in by_design.items()},
                n_samples=len(rows))


class MetricsLog:
    """One JSON object per line in <run_dir>/metrics.jsonl (appended across
    resumes), plus a short human-readable line on stdout."""

    def __init__(self, path):
        self.path = path

    def write(self, record, line=None):
        record = dict(record, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if line:
            print(line, flush=True)


def save_checkpoint(path, state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never leaves a truncated checkpoint


def run(cfg, resume=None):
    import memtrack
    from dataset import build_samples, load_unit, samples_in, split_designs, unit_paths

    run_dir = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    log = MetricsLog(os.path.join(run_dir, "metrics.jsonl"))
    cfg_hash = cfg.hash()

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if cfg.train.deterministic:
        # multithreaded CPU reductions sum in a run-dependent order (see config.py)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = resolve_device(cfg)

    samples = build_samples(cfg)
    split = split_designs(samples, cfg)
    if not split["val"]:
        raise ValueError("early stopping monitors val Spearman: the split needs >= 1 val design")
    sets = {k: samples_in(samples, v) for k, v in split.items()}
    units = [(i, p) for i, s in enumerate(sets["train"]) for p in unit_paths(s, cfg)]
    with open(os.path.join(run_dir, "config.resolved.json"), "w") as f:
        json.dump(dict(config=cfg.to_dict(), hash=cfg_hash), f, indent=1)
    with open(os.path.join(run_dir, "split.json"), "w") as f:
        json.dump(dict(split, samples={k: [(s["design"], s["variant"]) for s in v] for k, v in sets.items()}),
                  f, indent=1)

    first = load_unit(units[0][1], cfg.partition.enabled)
    in_dims = {nt: first[nt].x.shape[1] for nt in first.node_types}
    edge_types = list(first.edge_types)
    model = CongestionGNN(in_dims, edge_types, hidden_dim=cfg.model.hidden_dim,
                          num_layers=cfg.model.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    amp_dtype, use_scaler, amp_reason = resolve_amp(cfg, device, model, to_device(first, device), log)
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    start_epoch, micro_step, opt_steps = 0, 0, 0
    best, best_epoch, bad_epochs = -float("inf"), -1, 0
    if resume:
        path = os.path.join(run_dir, "last.pt") if resume == "auto" else resume
        ckpt = torch.load(path, weights_only=False)
        if ckpt["config_hash"] != cfg_hash:
            raise ValueError(f"checkpoint {path} was made with config {ckpt['config_hash']}, this run is "
                             f"{cfg_hash} - refusing to resume a different experiment (only train.epochs may change)")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, micro_step, opt_steps = ckpt["epoch"] + 1, ckpt["micro_step"], ckpt["opt_steps"]
        best, best_epoch, bad_epochs = ckpt["best"], ckpt["best_epoch"], ckpt["bad_epochs"]
        torch.set_rng_state(ckpt["rng"]["torch"]); np.random.set_state(ckpt["rng"]["numpy"])
        random.setstate(ckpt["rng"]["python"])
        log.write(dict(type="resume", from_checkpoint=path, next_epoch=start_epoch, best=best,
                       best_epoch=best_epoch, bad_epochs=bad_epochs, config_hash=cfg_hash),
                  f"resumed from {path}: next epoch {start_epoch}, best val spearman {best:.4f} @ {best_epoch}")
    else:
        log.write(dict(type="start", name=cfg.name, config_hash=cfg_hash, device=str(device), amp=amp_reason,
                       params=sum(p.numel() for p in model.parameters()),
                       deterministic=cfg.train.deterministic, torch_threads=torch.get_num_threads(),
                       split={k: v for k, v in split.items()},
                       n_samples={k: len(v) for k, v in sets.items()}, n_train_units=len(units),
                       torch=torch.__version__),
                  f"start {cfg.name} [{cfg_hash}] device={device} {amp_reason}; designs "
                  f"train/val/test = {len(split['train'])}/{len(split['val'])}/{len(split['test'])}, "
                  f"samples {len(sets['train'])}/{len(sets['val'])}/{len(sets['test'])}, {len(units)} train units")

    def checkpoint_state(epoch):
        return dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    epoch=epoch, micro_step=micro_step, opt_steps=opt_steps, best=best,
                    best_epoch=best_epoch, bad_epochs=bad_epochs, config=cfg.to_dict(), config_hash=cfg_hash,
                    rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate()),
                    # the keys app/viewer.py reads, so a best.pt loads there too
                    in_dims=in_dims, edge_types=edge_types, hidden_dim=cfg.model.hidden_dim,
                    num_layers=cfg.model.num_layers, trained_on=cfg.name)

    accum = cfg.train.accum_steps
    stopped = False
    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.perf_counter()
        model.train()
        # data order depends only on (seed, epoch) - identical with or without a resume
        order = torch.randperm(len(units), generator=torch.Generator().manual_seed(cfg.seed * 1_000_003 + epoch))
        sums = dict(loss=0.0, huber=0.0, rank=0.0)
        optimizer.zero_grad(set_to_none=True)
        for k, ui in enumerate(order.tolist()):
            group_start = (k // accum) * accum
            group_size = min(accum, len(order) - group_start)  # last group of an epoch may be short
            g = to_device(load_unit(units[ui][1], cfg.partition.enabled), device)
            with autocast_ctx(device, amp_dtype):
                out = model(g)
            core = g["gcell"].core
            pred, target = out[core].float(), g["gcell"].y_ratio[core]
            huber = F.smooth_l1_loss(pred, target)
            gen = torch.Generator().manual_seed(cfg.seed * 7_919 + micro_step)
            rank = sampled_ranking_loss(pred.cpu(), target.cpu(), cfg.train.rank_pairs, gen).to(pred.device)
            loss = huber + cfg.train.lambda_rank * rank
            scaler.scale(loss / group_size).backward()
            micro_step += 1
            sums["loss"] += loss.item(); sums["huber"] += huber.item(); sums["rank"] += rank.item()
            if k - group_start + 1 == group_size:
                if cfg.train.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
        train_time = time.perf_counter() - t0

        val = evaluate(model, sets["val"], cfg, device, amp_dtype)
        improved = val["spearman"] > best + cfg.early_stopping.min_delta
        if improved:
            best, best_epoch, bad_epochs = val["spearman"], epoch, 0
        else:
            bad_epochs += 1
        state = checkpoint_state(epoch)
        save_checkpoint(os.path.join(run_dir, "last.pt"), state)
        if improved:
            save_checkpoint(os.path.join(run_dir, "best.pt"), state)
        n = len(order)
        log.write(dict(type="epoch", epoch=epoch, micro_steps=micro_step, optimizer_steps=opt_steps,
                       lr=optimizer.param_groups[0]["lr"],
                       train={k: v / n for k, v in sums.items()} | dict(units=n),
                       val=val, improved=improved, best_val_spearman=best, best_epoch=best_epoch,
                       epochs_without_improvement=bad_epochs, train_time_s=train_time,
                       epoch_time_s=time.perf_counter() - t0, peak_mem_bytes=memtrack.peak_bytes()),
                  f"epoch {epoch:3d}  train loss {sums['loss'] / n:.4f} (huber {sums['huber'] / n:.4f}, "
                  f"rank {sums['rank'] / n:.4f})  val spearman {val['spearman']:.4f} "
                  f"(H {val['spearman_h']:.4f} V {val['spearman_v']:.4f})"
                  f"{'  *best*' if improved else f'  [{bad_epochs}/{cfg.early_stopping.patience}]'}  "
                  f"{time.perf_counter() - t0:.1f}s")
        if bad_epochs >= cfg.early_stopping.patience:
            log.write(dict(type="early_stop", epoch=epoch, best_epoch=best_epoch, best_val_spearman=best),
                      f"early stop at epoch {epoch}: no val spearman improvement for "
                      f"{bad_epochs} epochs (best {best:.4f} @ epoch {best_epoch})")
            stopped = True
            break

    best_ckpt = torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    test = evaluate(model, sets["test"], cfg, device, amp_dtype) if sets["test"] else None
    log.write(dict(type="final", best_epoch=best_ckpt["epoch"], best_val_spearman=best_ckpt["best"],
                   early_stopped=stopped, test=test),
              f"final: best epoch {best_ckpt['epoch']} val spearman {best_ckpt['best']:.4f}"
              + (f"; test spearman {test['spearman']:.4f} (H {test['spearman_h']:.4f} "
                 f"V {test['spearman_v']:.4f}) on {test['n_samples']} samples" if test else ""))
    return run_dir


def main():
    if "--config" not in sys.argv:
        overfit_main()  # Phase 7 CLI: python src/train.py <graph.pt> [...]
        return
    from config import load_config
    parser = argparse.ArgumentParser(description="Config-driven CongestionGNN training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override a config value (TOML syntax), repeatable")
    parser.add_argument("--resume", help="'auto' (<run_dir>/last.pt) or a checkpoint path")
    args = parser.parse_args()
    run(load_config(args.config, args.set), resume=args.resume)


if __name__ == "__main__":
    main()
