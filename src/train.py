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


def main():
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


if __name__ == "__main__":
    main()
