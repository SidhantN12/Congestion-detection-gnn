"""Phase 8 minimal viewer: sample selector, predicted vs ground truth
heatmaps on a shared colour scale, a signed difference panel, Spearman/
Kendall metrics, and a ranked top-20 hotspot table with click-to-zoom.

Model is loaded from the Phase 7 overfit-test checkpoint - this is a
pipeline-validation viewer, not a benchmark-trained model (see
notes/phase8-viewer.md). Predictions on samples other than the one it
was trained on are expected to generalise imperfectly; that's shown
plainly, not hidden.

Run: streamlit run app/viewer.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import CongestionGNN
from metrics import spearman_corr, kendall_tau

PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT_PATH = PROCESSED_DIR / "phase7_model_checkpoint.pt"

st.set_page_config(page_title="Congestion GNN Viewer", layout="wide")


@st.cache_resource
def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    model = CongestionGNN(
        ckpt["in_dims"], ckpt["edge_types"],
        hidden_dim=ckpt["hidden_dim"], num_layers=ckpt["num_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    trained_on_sample = Path(ckpt["trained_on"]).stem.removesuffix("_graph")
    return model, trained_on_sample


@st.cache_data
def list_samples():
    return sorted(p.stem.removesuffix("_graph") for p in PROCESSED_DIR.glob("bc*_bw*_s*_graph.pt"))


@st.cache_data
def load_graph(sample_name):
    return torch.load(PROCESSED_DIR / f"{sample_name}_graph.pt", weights_only=False)


def predict(_model, data):
    """Not cached (deliberately) - this is the operation the <1s gate
    times, so it must run fresh on every sample switch, not be served
    from a cache that would make the timing meaningless.
    """
    start = time.perf_counter()
    with torch.no_grad():
        pred = _model(data).numpy()
    elapsed = time.perf_counter() - start
    return pred, elapsed


model, trained_on_sample = load_model()
samples = list_samples()

st.title("Congestion GNN — Prediction Viewer")
st.caption(
    f"Model trained via Phase 7's single-sample overfit test on **{trained_on_sample}** "
    "— this is a pipeline-validation viewer, not a benchmark-trained model. Predictions "
    "on other samples are expected to generalise imperfectly; that's shown here plainly."
)

default_idx = samples.index(trained_on_sample) if trained_on_sample in samples else 0
selected = st.selectbox("Sample", samples, index=default_idx)

data = load_graph(selected)
pred, elapsed = predict(model, data)
target = data["gcell"].y_ratio.numpy()
grid_shape = tuple(data["gcell"].grid_shape.numpy())
ny, nx = grid_shape

st.caption(f"Re-predicted in **{elapsed * 1000:.1f} ms**"
           + (" ✅ under 1s" if elapsed < 1.0 else " ⚠️ over 1s"))

pred_combined = np.maximum(pred[:, 0], pred[:, 1]).reshape(grid_shape)
target_combined = np.maximum(target[:, 0], target[:, 1]).reshape(grid_shape)
diff = pred_combined - target_combined
vmax = max(pred_combined.max(), target_combined.max())
dmax = np.abs(diff).max()

col1, col2, col3 = st.columns(3)
fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
im1 = ax1.imshow(target_combined, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
ax1.set_title("Ground truth")
plt.colorbar(im1, ax=ax1, fraction=0.046)
col1.pyplot(fig1)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
im2 = ax2.imshow(pred_combined, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
ax2.set_title("Predicted")
plt.colorbar(im2, ax=ax2, fraction=0.046)
col2.pyplot(fig2)
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(4.5, 4.5))
im3 = ax3.imshow(diff, origin="lower", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
ax3.set_title("Predicted − actual")
plt.colorbar(im3, ax=ax3, fraction=0.046)
col3.pyplot(fig3)
plt.close(fig3)

sh = spearman_corr(pred[:, 0], target[:, 0])
sv = spearman_corr(pred[:, 1], target[:, 1])
kh = kendall_tau(pred[:, 0], target[:, 0])
kv = kendall_tau(pred[:, 1], target[:, 1])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Spearman (H)", f"{sh:.3f}")
m2.metric("Spearman (V)", f"{sv:.3f}")
m3.metric("Kendall τ (H)", f"{kh:.3f}")
m4.metric("Kendall τ (V)", f"{kv:.3f}")

st.subheader("Top-20 congestion hotspots (by ground truth)")
flat_idx = np.argsort(-target_combined.reshape(-1))[:20]
y_idx = flat_idx // nx
x_idx = flat_idx % nx
hotspot_df = pd.DataFrame({
    "rank": np.arange(1, 21),
    "gcell_index": flat_idx,
    "x": x_idx,
    "y": y_idx,
    "ground_truth": target_combined.reshape(-1)[flat_idx],
    "predicted": pred_combined.reshape(-1)[flat_idx],
    "abs_error": np.abs(pred_combined.reshape(-1)[flat_idx] - target_combined.reshape(-1)[flat_idx]),
})

event = st.dataframe(
    hotspot_df, hide_index=True, width="stretch",
    on_select="rerun", selection_mode="single-row",
)

st.caption("Click a row above to zoom into that GCell's neighbourhood.")

selected_rows = event.selection.rows if event and event.selection else []
if selected_rows:
    row = hotspot_df.iloc[selected_rows[0]]
    zx, zy = int(row["x"]), int(row["y"])
    pad = 3
    x_lo, x_hi = max(0, zx - pad), min(nx, zx + pad + 1)
    y_lo, y_hi = max(0, zy - pad), min(ny, zy + pad + 1)

    zcol1, zcol2 = st.columns(2)
    fig_z1, ax_z1 = plt.subplots(figsize=(4, 4))
    ax_z1.imshow(target_combined[y_lo:y_hi, x_lo:x_hi], origin="lower", cmap="inferno",
                 vmin=0, vmax=vmax, extent=[x_lo, x_hi, y_lo, y_hi])
    ax_z1.scatter([zx + 0.5], [zy + 0.5], marker="x", color="cyan", s=100)
    ax_z1.set_title(f"Ground truth zoom around ({zx},{zy})")
    zcol1.pyplot(fig_z1)
    plt.close(fig_z1)

    fig_z2, ax_z2 = plt.subplots(figsize=(4, 4))
    ax_z2.imshow(pred_combined[y_lo:y_hi, x_lo:x_hi], origin="lower", cmap="inferno",
                 vmin=0, vmax=vmax, extent=[x_lo, x_hi, y_lo, y_hi])
    ax_z2.scatter([zx + 0.5], [zy + 0.5], marker="x", color="cyan", s=100)
    ax_z2.set_title(f"Predicted zoom around ({zx},{zy})")
    zcol2.pyplot(fig_z2)
    plt.close(fig_z2)
