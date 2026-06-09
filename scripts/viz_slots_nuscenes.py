"""
nuScenes Slot Visualization: SlotAttention vs GATST
====================================================
For each nuScenes scene, produces 3 files:
  1. slots_XXXX_SA.png    — full-size grid, one panel per SA slot (heatmap overlay)
  2. slots_XXXX_GATST.png — same for GATST
  3. slots_XXXX_compare.png — side-by-side segmaps + loss curve

Each slot panel: attention heatmap (jet) blended onto the real image.
High-attention patches = bright red/yellow.  Low = dark blue → transparent.
"""

import sys, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, "/work/DriveWeaver/videosaur")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as MplNorm
import matplotlib.cm as cm
from PIL import Image
import timm
from torchvision import transforms
from videosaur.modules import groupers

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
INP_DIM     = 384
SLOT_DIM    = 128
N_SLOTS     = 11
N_ITERS     = 3
N_HEADS     = 4
PATCH_SIDE  = 16
N_PATCHES   = PATCH_SIDE * PATCH_SIDE
TRAIN_STEPS = 800
LR          = 3e-4
IMG_SIZE    = 224

IMG_PATHS = [
    ("/work/dataset_annotations/nuscenes_scene_context_qa_DRYRUN/scene-0402/sample_012654.jpg", "0402"),
    ("/work/dataset_annotations/nuscenes_scene_context_qa_DRYRUN/scene-0637/sample_019684.jpg", "0637"),
    ("/work/dataset_annotations/nuscenes_scene_context_qa_DRYRUN/scene-0716/sample_022496.jpg", "0716"),
]
OUT_DIR = "/work/DriveWeaver/outputs/viz_nuscenes_slots"
os.makedirs(OUT_DIR, exist_ok=True)

SLOT_COLORS_HEX = [
    "#FF2222","#FF9900","#FFDD00","#22CC22","#00DDFF",
    "#2255FF","#BB00FF","#FF55CC","#00BB77","#AA4400","#AAAAAA",
]
SLOT_COLORS_RGB = [
    tuple(int(h[i:i+2],16)/255. for i in (1,3,5)) for h in SLOT_COLORS_HEX
]
NORMALIZE = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])


# ── Image loading ─────────────────────────────────────────────────────────────
def load_nuscenes(path):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    img  = img.crop((0, int(H*0.10), W, int(H*0.85)))
    img_sq = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    tensor = NORMALIZE(transforms.ToTensor()(img_sq)).unsqueeze(0).to(DEVICE)
    return tensor, np.array(img_sq).astype(np.float32)/255.0


# ── DINOv2 features ───────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(tensor):
    m = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                           num_classes=0, img_size=IMG_SIZE).to(DEVICE).eval()
    feats = m.forward_features(tensor)[:,1:,:]
    del m; return feats


# ── Model wrapper with learnable slot embeddings ──────────────────────────────
class GrouperWithSlots(nn.Module):
    def __init__(self, grouper):
        super().__init__()
        self.grouper = grouper
        self.slots   = nn.Parameter(torch.randn(1, N_SLOTS, SLOT_DIM)*0.02)
    def forward(self, features):
        s = self.slots.expand(features.shape[0],-1,-1)
        return self.grouper(s, features)


# ── Training ──────────────────────────────────────────────────────────────────
def train(model_cls, model_kwargs, features, seed=42):
    torch.manual_seed(seed)
    g      = model_cls(**model_kwargs).to(DEVICE)
    model  = GrouperWithSlots(g).to(DEVICE)
    dec    = nn.Linear(SLOT_DIM, INP_DIM).to(DEVICE)
    opt    = Adam(list(model.parameters())+list(dec.parameters()), lr=LR)
    fn     = F.normalize(features, dim=-1)
    losses = []
    for step in range(TRAIN_STEPS):
        out   = model(features)
        masks = out["masks"]
        if model_cls == groupers.SlotAttention:
            w = masks / (masks.sum(-1, keepdim=True)+1e-8)
        else:
            w = masks
        recon = torch.einsum("bkn,bkd->bnd", w, dec(out["slots"]))
        loss  = F.mse_loss(recon, fn)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"    {model_cls.__name__:20s} "
          f"step1={losses[0]:.4f} step400={losses[399]:.4f} final={losses[-1]:.4f}")
    return model, losses


# ── Get slot masks ────────────────────────────────────────────────────────────
def get_masks(model, model_cls, features):
    with torch.no_grad():
        out   = model(features)
        masks = out["masks"].squeeze(0).cpu()  # [K, N]
    if model_cls == groupers.SlotAttention:
        # softmax was over K → renorm per slot over N
        masks = masks / (masks.sum(-1, keepdim=True)+1e-8)
    # GATST already has softmax over N
    return masks   # [K, N], each row sums to 1


# ── Build attention map [H, W] from 1D mask ───────────────────────────────────
def mask_to_heatmap(mask_1d, img_size=IMG_SIZE):
    """
    mask_1d : [N] — values in [0,1] (normalised per-slot)
    Returns  : [H, W] float in [0,1]  (bilinear-upsampled)
    """
    m2d = mask_1d.reshape(1, 1, PATCH_SIDE, PATCH_SIDE).float()
    up  = F.interpolate(m2d, size=(img_size, img_size),
                         mode="bilinear", align_corners=False)
    arr = up.squeeze().numpy()
    # Normalise to 0-1 per slot so dim slots are still visible
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


# ── Overlay: heatmap blended on image ─────────────────────────────────────────
def heatmap_overlay(img_np, attn_map, alpha=0.60, cmap="jet"):
    """
    img_np   : [H,W,3]
    attn_map : [H,W] 0-1
    Returns  : [H,W,3]
    """
    colormap = cm.get_cmap(cmap)
    coloured = colormap(attn_map)[..., :3]          # [H,W,3] jet colours
    blended  = (1-alpha) * img_np + alpha * coloured
    return blended.clip(0,1)


# ── Slot image with border ────────────────────────────────────────────────────
def render_slot(img_np, mask_1d, slot_idx):
    attn = mask_to_heatmap(mask_1d)
    return heatmap_overlay(img_np, attn, alpha=0.65)


# ── Hard segmentation map ─────────────────────────────────────────────────────
def segmap(masks_kn, img_np):
    argmax = masks_kn.argmax(0).numpy()  # [N]
    seg = np.zeros((PATCH_SIDE, PATCH_SIDE, 3), dtype=np.float32)
    for i, k in enumerate(argmax):
        seg[i//PATCH_SIDE, i%PATCH_SIDE] = SLOT_COLORS_RGB[k%len(SLOT_COLORS_RGB)]
    seg_up = F.interpolate(
        torch.from_numpy(seg).permute(2,0,1).unsqueeze(0),
        size=(IMG_SIZE, IMG_SIZE), mode="nearest"
    ).squeeze().permute(1,2,0).numpy()
    return (0.50*seg_up + 0.50*img_np).clip(0,1)


# ── Figure 1: all slots for one model (large) ─────────────────────────────────
def make_slot_grid(img_np, masks, model_name, scene_id, color_label, losses):
    """
    4 rows × 3 cols = 12 panels (11 slots + 1 overview segmap)
    Each panel: heatmap overlay at full resolution
    """
    K = N_SLOTS
    cols = 4
    rows = 3   # ceil(K+1 / cols)
    panel_size = 3.4   # inches per panel

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols*panel_size, rows*panel_size+0.9),
                              facecolor="#111118")
    fig.suptitle(
        f"Scene {scene_id}  ·  {model_name}  ·  {K} slots  ·  {TRAIN_STEPS} steps\n"
        f"Each panel: attention heatmap overlaid on image  "
        f"(red=high attention, blue=low)   "
        f"Final MSE = {losses[-1]:.5f}",
        color="white", fontsize=12, fontweight="bold", y=0.99
    )

    axes_flat = axes.flatten()

    # First panel: segmentation map (argmax per patch)
    ax0 = axes_flat[0]
    ax0.imshow(segmap(masks, img_np))
    ax0.set_title("Segmentation\n(argmax slot per patch)",
                  color=color_label, fontsize=9, pad=4)
    ax0.axis("off")
    for sp in ax0.spines.values(): sp.set_edgecolor("white"); sp.set_linewidth(2)
    ax0.set_frame_on(True)

    # Remaining panels: one per slot
    for k in range(K):
        ax = axes_flat[k+1]
        panel = render_slot(img_np, masks[k], k)
        ax.imshow(panel)
        hex_c = SLOT_COLORS_HEX[k%len(SLOT_COLORS_HEX)]
        # Title: slot index + peak attention value
        peak = masks[k].max().item()
        ax.set_title(f"Slot {k}   (peak={peak:.4f})", color=hex_c, fontsize=9, pad=3)
        ax.axis("off")
        for sp in ax.spines.values(): sp.set_edgecolor(hex_c); sp.set_linewidth(3)
        ax.set_frame_on(True)

    # Hide last panel (12th = unused)
    axes_flat[-1].set_visible(False)

    # Colourbar legend
    sm = plt.cm.ScalarMappable(cmap="jet", norm=MplNorm(0,1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat[-1], fraction=0.9, pad=0.02)
    cbar.set_label("Normalised attention", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
    axes_flat[-1].set_visible(True)
    axes_flat[-1].axis("off")

    plt.tight_layout(rect=[0,0,1,0.96])
    return fig


# ── Figure 2: side-by-side comparison (seg maps + loss) ──────────────────────
def make_compare_fig(img_np, sa_masks, gt_masks, sa_losses, gt_losses, scene_id):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), facecolor="#111118")
    fig.suptitle(
        f"nuScenes Scene {scene_id}  ·  SlotAttention vs GATST  ·  Segmentation Comparison",
        color="white", fontsize=13, fontweight="bold"
    )

    axes[0].imshow(img_np)
    axes[0].set_title("Original", color="white", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(segmap(sa_masks, img_np))
    axes[1].set_title(f"SlotAttention Segmentation\nMSE={sa_losses[-1]:.5f}",
                      color="#FF7777", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(segmap(gt_masks, img_np))
    axes[2].set_title(f"GATST Segmentation\nMSE={gt_losses[-1]:.5f}",
                      color="#77AAFF", fontsize=11)
    axes[2].axis("off")

    axes[3].set_facecolor("#0a0a14")
    s = list(range(1, TRAIN_STEPS+1))
    axes[3].plot(s, sa_losses, color="#FF7777", lw=2, label="SlotAttention")
    axes[3].plot(s, gt_losses, color="#77AAFF", lw=2, label="GATST")
    axes[3].set_title("Reconstruction Loss", color="white", fontsize=11)
    axes[3].set_xlabel("Step", color="#aaa"); axes[3].set_ylabel("MSE", color="#aaa")
    axes[3].tick_params(colors="#aaa")
    axes[3].legend(facecolor="#0a0a14", labelcolor="white", fontsize=10)
    axes[3].grid(alpha=0.2)
    for sp in axes[3].spines.values(): sp.set_edgecolor("#333")

    for ax in axes[:3]:
        for sp in ax.spines.values(): sp.set_edgecolor("#444"); sp.set_linewidth(1.5)
        ax.set_frame_on(True)

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sa_kw    = dict(inp_dim=INP_DIM, slot_dim=SLOT_DIM, n_iters=N_ITERS)
    gatst_kw = dict(inp_dim=INP_DIM, slot_dim=SLOT_DIM, n_heads=N_HEADS, n_iters=N_ITERS, n_patches=N_PATCHES, pos_scale=3.0)

    for img_path, scene_id in IMG_PATHS:
        print(f"\n{'='*60}  Scene {scene_id}  {'='*5}")

        tensor, img_np = load_nuscenes(img_path)
        print("  Extracting DINOv2 features...")
        features = extract_features(tensor)

        print(f"  Training SlotAttention ({TRAIN_STEPS} steps)...")
        sa_model, sa_losses = train(groupers.SlotAttention, sa_kw, features)
        print(f"  Training GATST ({TRAIN_STEPS} steps)...")
        gt_model, gt_losses = train(groupers.GATST, gatst_kw, features)

        sa_masks = get_masks(sa_model, groupers.SlotAttention, features)
        gt_masks = get_masks(gt_model, groupers.GATST, features)

        print(f"  SA    peak per slot: {[f'{sa_masks[k].max():.4f}' for k in range(N_SLOTS)]}")
        print(f"  GATST peak per slot: {[f'{gt_masks[k].max():.4f}' for k in range(N_SLOTS)]}")

        print("  Rendering SA figure...")
        fig1 = make_slot_grid(img_np, sa_masks, "SlotAttention", scene_id, "#FF7777", sa_losses)
        fig1.savefig(f"{OUT_DIR}/slots_{scene_id}_SA.png", dpi=150, bbox_inches="tight",
                     facecolor=fig1.get_facecolor())
        plt.close(fig1)

        print("  Rendering GATST figure...")
        fig2 = make_slot_grid(img_np, gt_masks, "GATST", scene_id, "#77AAFF", gt_losses)
        fig2.savefig(f"{OUT_DIR}/slots_{scene_id}_GATST.png", dpi=150, bbox_inches="tight",
                     facecolor=fig2.get_facecolor())
        plt.close(fig2)

        print("  Rendering comparison figure...")
        fig3 = make_compare_fig(img_np, sa_masks, gt_masks, sa_losses, gt_losses, scene_id)
        fig3.savefig(f"{OUT_DIR}/slots_{scene_id}_compare.png", dpi=150, bbox_inches="tight",
                     facecolor=fig3.get_facecolor())
        plt.close(fig3)

        print(f"  Saved: slots_{scene_id}_SA.png  |  slots_{scene_id}_GATST.png  |  slots_{scene_id}_compare.png")

    print(f"\nAll files saved to {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")
