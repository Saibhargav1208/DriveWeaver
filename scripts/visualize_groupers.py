"""
Visualization: SlotAttention vs GATST on real images.

For each image:
  1. Extract DINOv2-small patch features  (14x14 grid = 196 patches)
  2. Run SlotAttention and GATST (untrained, same random init seed)
  3. Train each for 400 steps to segment the image (MSE reconstruction of DINOv2 feats)
  4. Produce a figure showing:
       Row 0: original image
       Row 1: SA slot masks overlaid (K coloured masks)
       Row 2: GATST slot masks overlaid (K coloured masks)
       Row 3: SA segmentation map (hard assignment)
       Row 4: GATST segmentation map (hard assignment)

Also produces:
  - Convergence curve (MSE vs steps)
  - Slot entropy over training (shows collapse / utilisation)
  - Attention map grid for each slot
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/work/DriveWeaver/videosaur")

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import timm
from torchvision import transforms

from videosaur.modules import groupers

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
INP_DIM   = 384    # DINOv2-small
SLOT_DIM  = 64
N_SLOTS   = 8
N_ITERS   = 3
N_HEADS   = 4
PATCH_H   = 16     # 224px / 14px-patch = 16 patches per side
PATCH_W   = 16
N_PATCHES = PATCH_H * PATCH_W   # 256
TRAIN_STEPS = 400
LR          = 5e-4

IMG_PATHS = [
    "/work/DriveWeaver/videosaur/docs/static/images/test.jpg",
    "/work/DriveWeaver/videosaur/docs/static/images/test2.jpg",
]
OUT_DIR = "/work/DriveWeaver/outputs/viz_groupers"

import os
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette (K distinct colours) ─────────────────────────────────────────────
PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff",
]
PALETTE_RGB = [tuple(int(h[i:i+2], 16)/255 for i in (1,3,5)) for h in PALETTE]


# ── DINOv2 feature extractor ─────────────────────────────────────────────────
class DINOv2Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "vit_small_patch14_dinov2", pretrained=True, num_classes=0, img_size=224
        )

    @torch.no_grad()
    def forward(self, x):
        # returns patch tokens [B, N, 384]  (no CLS)
        feats = self.model.forward_features(x)
        # timm ViT: feats shape [B, N+1, D] — strip CLS
        return feats[:, 1:, :]   # [B, 256, 384]


NORMALIZE = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0).to(DEVICE)
    img_small = img.resize((224, 224), Image.LANCZOS)
    return tensor, img_small


# ── Training (per-image slot learning) ───────────────────────────────────────
def train_grouper(model_cls, model_kwargs, features, n_steps=TRAIN_STEPS, seed=0):
    """
    features: [1, N, D] — fixed DINOv2 features for one image.
    Train the grouper + linear decoder to reconstruct the features.
    Returns (model, decoder, loss_history, entropy_history).
    """
    torch.manual_seed(seed)
    model   = model_cls(**model_kwargs).to(DEVICE)
    decoder = nn.Linear(SLOT_DIM, INP_DIM).to(DEVICE)
    opt     = Adam(list(model.parameters()) + list(decoder.parameters()), lr=LR)

    feat_norm = F.normalize(features, dim=-1)  # [1, N, D]
    loss_hist = []
    ent_hist  = []

    for step in range(n_steps):
        torch.manual_seed(step + seed * 10000)
        slots_in = torch.randn(1, N_SLOTS, SLOT_DIM, device=DEVICE)

        out   = model(slots_in, features)
        slots = out["slots"]   # [1, K, slot_dim]
        masks = out["masks"]   # [1, K, N]

        # Per-slot normalised weights for reconstruction
        if model_cls == groupers.SlotAttention:
            # SA: masks = softmax over K → to reconstruct patches, re-normalise over N
            w = masks / (masks.sum(-1, keepdim=True) + 1e-8)
        else:
            w = masks   # GATST: already softmax over N

        slot_dec = decoder(slots)   # [1, K, D]
        recon    = torch.einsum("bkn,bkd->bnd", w, slot_dec)  # [1, N, D]
        loss     = F.mse_loss(recon, feat_norm)
        opt.zero_grad(); loss.backward(); opt.step()

        loss_hist.append(loss.item())

        # Entropy of slot attention (over patches)
        with torch.no_grad():
            if model_cls == groupers.SlotAttention:
                p = masks / (masks.sum(-1, keepdim=True) + 1e-8)
            else:
                p = masks
            ent = -(p * (p + 1e-8).log()).sum(-1).mean().item()
        ent_hist.append(ent)

    return model, decoder, loss_hist, ent_hist


def get_masks(model, model_cls, features, n_seeds=16):
    """
    Run model with multiple random slot inits, return averaged soft masks.
    masks_out: [K, N]  (averaged over seeds, normalised over N per slot)
    """
    all_masks = []
    with torch.no_grad():
        for s in range(n_seeds):
            torch.manual_seed(s + 9000)
            slots_in = torch.randn(1, N_SLOTS, SLOT_DIM, device=DEVICE)
            out = model(slots_in, features)
            masks = out["masks"].squeeze(0)  # [K, N]
            if model_cls == groupers.SlotAttention:
                masks = masks / (masks.sum(-1, keepdim=True) + 1e-8)
            all_masks.append(masks)
    masks_avg = torch.stack(all_masks).mean(0)   # [K, N]
    return masks_avg


# ── Visualisation helpers ─────────────────────────────────────────────────────
def masks_to_rgb_overlay(img_np, masks_kn, alpha=0.55):
    """
    img_np:   [H, W, 3] float32 in [0,1]
    masks_kn: [K, N]  — each row is a soft weight over patches (normalised)
    Returns:  [H, W, 3] blended overlay
    """
    K, N = masks_kn.shape
    ph, pw = PATCH_H, PATCH_W
    H, W = img_np.shape[:2]

    overlay = np.zeros((ph, pw, 3), dtype=np.float32)
    for k in range(K):
        m = masks_kn[k].reshape(ph, pw).numpy()
        colour = np.array(PALETTE_RGB[k % len(PALETTE_RGB)])
        overlay += m[:, :, None] * colour[None, None, :]  # [ph,pw,3]

    # Upsample overlay to image size
    overlay_img = Image.fromarray((overlay * 255).clip(0,255).astype(np.uint8))
    overlay_img = overlay_img.resize((W, H), Image.NEAREST)
    overlay_np  = np.array(overlay_img).astype(np.float32) / 255.

    blended = (1 - alpha) * img_np + alpha * overlay_np
    return blended.clip(0, 1)


def masks_to_segmap(masks_kn):
    """
    Hard assignment: each patch gets the colour of its argmax slot.
    Returns [PH, PW, 3] RGB image (patch resolution).
    """
    K, N = masks_kn.shape
    argmax = masks_kn.argmax(dim=0).numpy()   # [N]
    seg = np.array([PALETTE_RGB[i % len(PALETTE_RGB)] for i in argmax], dtype=np.float32)
    return seg.reshape(PATCH_H, PATCH_W, 3)


def per_slot_heatmaps(masks_kn):
    """Return list of K greyscale heatmaps (each PATCH_H x PATCH_W)."""
    K, N = masks_kn.shape
    out = []
    for k in range(K):
        m = masks_kn[k].reshape(PATCH_H, PATCH_W).numpy()
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        out.append(m)
    return out


def slot_entropy_vals(masks_kn):
    """Mean entropy over slots (higher = more spread attention per slot)."""
    p = masks_kn / (masks_kn.sum(-1, keepdim=True) + 1e-8)
    ent = -(p * (p + 1e-8).log()).sum(-1)
    return ent.numpy()


# ── Main figure builder ───────────────────────────────────────────────────────
def make_figure(img_path, img_idx):
    print(f"\n{'='*60}")
    print(f"Processing image {img_idx+1}: {img_path}")
    print(f"{'='*60}")

    tensor, img_pil = load_image(img_path)
    img_np = np.array(img_pil).astype(np.float32) / 255.

    # Extract DINOv2 features
    print("  Extracting DINOv2-small features...")
    extractor = DINOv2Extractor().to(DEVICE).eval()
    with torch.no_grad():
        features = extractor(tensor)   # [1, 196, 384]
    del extractor

    sa_kwargs    = dict(inp_dim=INP_DIM, slot_dim=SLOT_DIM, n_iters=N_ITERS)
    gatst_kwargs = dict(inp_dim=INP_DIM, slot_dim=SLOT_DIM, n_heads=N_HEADS, n_iters=N_ITERS)

    print(f"  Training SlotAttention ({TRAIN_STEPS} steps)...")
    sa_model, _, sa_loss, sa_ent = train_grouper(
        groupers.SlotAttention, sa_kwargs, features, seed=42
    )
    print(f"    final MSE: {sa_loss[-1]:.5f}")

    print(f"  Training GATST ({TRAIN_STEPS} steps)...")
    gatst_model, _, gatst_loss, gatst_ent = train_grouper(
        groupers.GATST, gatst_kwargs, features, seed=42
    )
    print(f"    final MSE: {gatst_loss[-1]:.5f}")

    # Get masks
    sa_masks    = get_masks(sa_model,    groupers.SlotAttention, features).cpu()
    gatst_masks = get_masks(gatst_model, groupers.GATST,         features).cpu()

    sa_ent_vals    = slot_entropy_vals(sa_masks)
    gatst_ent_vals = slot_entropy_vals(gatst_masks)

    # ── Figure 1: Main comparison ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(
        f"SlotAttention vs GATST — Image {img_idx+1}\n"
        f"DINOv2-small features, {N_SLOTS} slots, {TRAIN_STEPS} training steps",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Col 0: Original image
    axes[0, 0].imshow(img_np); axes[0, 0].set_title("Original Image", fontsize=11)
    axes[0, 0].axis("off")

    # Col 1: SA overlay
    sa_overlay = masks_to_rgb_overlay(img_np, sa_masks)
    axes[0, 1].imshow(sa_overlay); axes[0, 1].set_title("SlotAttention\nSoft Mask Overlay", fontsize=11)
    axes[0, 1].axis("off")

    # Col 2: GATST overlay
    gatst_overlay = masks_to_rgb_overlay(img_np, gatst_masks)
    axes[0, 2].imshow(gatst_overlay); axes[0, 2].set_title("GATST\nSoft Mask Overlay", fontsize=11)
    axes[0, 2].axis("off")

    # Row 1: Hard segmentation maps
    sa_seg    = masks_to_segmap(sa_masks)
    gatst_seg = masks_to_segmap(gatst_masks)

    axes[1, 0].axis("off")  # blank

    sa_seg_up = np.array(Image.fromarray((sa_seg*255).astype(np.uint8)).resize((224,224), Image.NEAREST)).astype(np.float32)/255
    axes[1, 1].imshow(sa_seg_up)
    axes[1, 1].set_title(
        f"SlotAttention Hard Segmentation\n"
        f"Mean slot entropy: {sa_ent_vals.mean():.2f} (max={math.log(N_PATCHES):.2f})",
        fontsize=10
    )
    axes[1, 1].axis("off")

    gatst_seg_up = np.array(Image.fromarray((gatst_seg*255).astype(np.uint8)).resize((224,224), Image.NEAREST)).astype(np.float32)/255
    axes[1, 2].imshow(gatst_seg_up)
    axes[1, 2].set_title(
        f"GATST Hard Segmentation\n"
        f"Mean slot entropy: {gatst_ent_vals.mean():.2f} (max={math.log(N_PATCHES):.2f})",
        fontsize=10
    )
    axes[1, 2].axis("off")

    # Row 2: Convergence + Entropy plots
    steps = list(range(1, TRAIN_STEPS + 1))
    axes[2, 0].plot(steps, sa_loss,    color="#e6194b", label="SlotAttention", linewidth=2)
    axes[2, 0].plot(steps, gatst_loss, color="#4363d8", label="GATST",         linewidth=2)
    axes[2, 0].set_title("Reconstruction MSE vs Training Steps", fontsize=10)
    axes[2, 0].set_xlabel("Step"); axes[2, 0].set_ylabel("MSE")
    axes[2, 0].legend(); axes[2, 0].grid(alpha=0.3)

    axes[2, 1].plot(steps, sa_ent,    color="#e6194b", label="SlotAttention", linewidth=2)
    axes[2, 1].plot(steps, gatst_ent, color="#4363d8", label="GATST",         linewidth=2)
    axes[2, 1].set_title("Mean Slot Entropy vs Training Steps\n(higher = more spread, less collapse)", fontsize=10)
    axes[2, 1].set_xlabel("Step"); axes[2, 1].set_ylabel("Entropy (nats)")
    axes[2, 1].legend(); axes[2, 1].grid(alpha=0.3)

    # Per-slot entropy bar chart
    x = np.arange(N_SLOTS)
    w = 0.35
    axes[2, 2].bar(x - w/2, sa_ent_vals,    width=w, color="#e6194b", alpha=0.85, label="SlotAttention")
    axes[2, 2].bar(x + w/2, gatst_ent_vals, width=w, color="#4363d8", alpha=0.85, label="GATST")
    axes[2, 2].axhline(math.log(N_PATCHES), color="black", linestyle="--", linewidth=1, label=f"Max ({math.log(N_PATCHES):.2f})")
    axes[2, 2].set_title("Per-Slot Entropy (Attention Spread)", fontsize=10)
    axes[2, 2].set_xlabel("Slot Index"); axes[2, 2].set_ylabel("Entropy (nats)")
    axes[2, 2].set_xticks(x); axes[2, 2].legend(fontsize=8); axes[2, 2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = f"{OUT_DIR}/comparison_img{img_idx+1}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ── Figure 2: Per-slot heatmap grid ──────────────────────────────────────
    sa_hmaps    = per_slot_heatmaps(sa_masks)
    gatst_hmaps = per_slot_heatmaps(gatst_masks)

    fig2, axes2 = plt.subplots(2, N_SLOTS, figsize=(2.5 * N_SLOTS, 6))
    fig2.suptitle(
        f"Per-Slot Attention Heatmaps — Image {img_idx+1}\n"
        f"Top: SlotAttention   Bottom: GATST",
        fontsize=13, fontweight="bold"
    )

    for k in range(N_SLOTS):
        # SA row
        axes2[0, k].imshow(sa_hmaps[k], cmap="hot", vmin=0, vmax=1)
        axes2[0, k].set_title(f"SA\nSlot {k}", fontsize=8)
        axes2[0, k].axis("off")
        # Mark dead slots (max activation < 0.1)
        if sa_hmaps[k].max() < 0.1:
            axes2[0, k].set_facecolor("#ffcccc")
            axes2[0, k].set_title(f"SA\nSlot {k}\n[DEAD]", fontsize=8, color="red")

        # GATST row
        axes2[1, k].imshow(gatst_hmaps[k], cmap="hot", vmin=0, vmax=1)
        axes2[1, k].set_title(f"GATST\nSlot {k}", fontsize=8)
        axes2[1, k].axis("off")
        if gatst_hmaps[k].max() < 0.1:
            axes2[1, k].set_facecolor("#ccccff")
            axes2[1, k].set_title(f"GATST\nSlot {k}\n[DEAD]", fontsize=8, color="blue")

    plt.tight_layout()
    out_path2 = f"{OUT_DIR}/slot_heatmaps_img{img_idx+1}.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_path2}")

    # ── Figure 3: Competitive bottleneck illustration ─────────────────────────
    # Show per-patch softmax distributions to illustrate SA competition vs GATST
    with torch.no_grad():
        torch.manual_seed(9000)
        slots_in = torch.randn(1, N_SLOTS, SLOT_DIM, device=DEVICE)
        sa_out    = sa_model(slots_in, features)
        gatst_out = gatst_model(slots_in, features)

    # SA: masks = softmax over slots (competitive)
    sa_raw  = sa_out["masks"].squeeze(0)    # [K, N]  softmax over K
    # GATST: masks = softmax over patches  → transpose view
    gatst_raw = gatst_out["masks"].squeeze(0)  # [K, N]  softmax over N

    # Show patch ownership probability (for SA this is directly sa_raw[k,n] = P(slot k owns patch n))
    # For GATST: P(slot k attends to patch n) = gatst_raw[k,n]
    # Compare: for SA max_k(sa_raw[k,n]) shows how "exclusive" the patch is
    # For GATST max_k(gatst_raw[k,n]) after normalising over K

    sa_ownership    = sa_raw.cpu().numpy()             # [K,N]  already sums to 1 over K
    gatst_norm      = (gatst_raw / (gatst_raw.sum(0, keepdim=True) + 1e-8)).cpu().numpy()  # [K,N]

    fig3, axes3 = plt.subplots(2, 4, figsize=(18, 8))
    fig3.suptitle(
        f"Competitive Bottleneck Analysis — Image {img_idx+1}\n"
        "SA: patches compete for slots (sum over slots = 1)  |  "
        "GATST: slots freely attend to patches (no competition)",
        fontsize=12, fontweight="bold"
    )

    # Max ownership per patch → exclusivity map
    sa_excl    = sa_ownership.max(axis=0).reshape(PATCH_H, PATCH_W)
    gatst_excl = gatst_norm.max(axis=0).reshape(PATCH_H, PATCH_W)

    axes3[0, 0].imshow(img_np)
    axes3[0, 0].set_title("Original", fontsize=10); axes3[0, 0].axis("off")

    im = axes3[0, 1].imshow(sa_excl, cmap="RdYlGn", vmin=0, vmax=1)
    axes3[0, 1].set_title(f"SA: Max Slot Ownership per Patch\n(mean={sa_excl.mean():.3f}, 1=fully exclusive)", fontsize=9)
    axes3[0, 1].axis("off")
    fig3.colorbar(im, ax=axes3[0, 1], fraction=0.046)

    im = axes3[0, 2].imshow(gatst_excl, cmap="RdYlGn", vmin=0, vmax=1)
    axes3[0, 2].set_title(f"GATST: Max Slot Attendance per Patch\n(mean={gatst_excl.mean():.3f}, 1=fully exclusive)", fontsize=9)
    axes3[0, 2].axis("off")
    fig3.colorbar(im, ax=axes3[0, 2], fraction=0.046)

    # Winner map: which model gives more exclusive assignment
    diff = gatst_excl - sa_excl
    im = axes3[0, 3].imshow(diff, cmap="bwr", vmin=-0.5, vmax=0.5)
    axes3[0, 3].set_title("Difference (GATST - SA)\nBlue=SA more exclusive, Red=GATST more exclusive", fontsize=9)
    axes3[0, 3].axis("off")
    fig3.colorbar(im, ax=axes3[0, 3], fraction=0.046)

    # Row 2: Distribution of max-ownership values (histogram)
    axes3[1, 0].hist(sa_excl.flatten(),    bins=20, color="#e6194b", alpha=0.7, label="SlotAttention")
    axes3[1, 0].hist(gatst_excl.flatten(), bins=20, color="#4363d8", alpha=0.7, label="GATST")
    axes3[1, 0].set_title("Distribution of Max Patch Ownership\n(right = more exclusive/crisp)", fontsize=9)
    axes3[1, 0].set_xlabel("Max ownership value"); axes3[1, 0].set_ylabel("Patch count")
    axes3[1, 0].legend(); axes3[1, 0].grid(alpha=0.3)

    # How many slots own each patch? (SA: always sums to 1 → can be fragmented)
    sa_fragmentation    = (sa_ownership > 0.1).sum(axis=0).reshape(PATCH_H, PATCH_W)
    gatst_fragmentation = (gatst_norm   > 0.1).sum(axis=0).reshape(PATCH_H, PATCH_W)

    im = axes3[1, 1].imshow(sa_fragmentation, cmap="YlOrRd", vmin=1, vmax=N_SLOTS)
    axes3[1, 1].set_title(f"SA: # Slots Significantly Owning Patch\n(mean={sa_fragmentation.mean():.2f}, lower=crisper)", fontsize=9)
    axes3[1, 1].axis("off")
    fig3.colorbar(im, ax=axes3[1, 1], fraction=0.046)

    im = axes3[1, 2].imshow(gatst_fragmentation, cmap="YlOrRd", vmin=1, vmax=N_SLOTS)
    axes3[1, 2].set_title(f"GATST: # Slots Significantly Attending Patch\n(mean={gatst_fragmentation.mean():.2f}, lower=crisper)", fontsize=9)
    axes3[1, 2].axis("off")
    fig3.colorbar(im, ax=axes3[1, 2], fraction=0.046)

    # Slot utilisation: how many patches does each slot attend to?
    sa_coverage    = (sa_ownership > 1.0/N_SLOTS).sum(axis=1)   # patches where this slot is dominant
    gatst_coverage = (gatst_norm   > 1.0/N_SLOTS).sum(axis=1)
    axes3[1, 3].bar(np.arange(N_SLOTS) - 0.2, sa_coverage,    0.4, color="#e6194b", alpha=0.85, label="SlotAttention")
    axes3[1, 3].bar(np.arange(N_SLOTS) + 0.2, gatst_coverage, 0.4, color="#4363d8", alpha=0.85, label="GATST")
    axes3[1, 3].set_title("# Patches Each Slot is Dominant In\n(dead slot = bar near 0)", fontsize=9)
    axes3[1, 3].set_xlabel("Slot"); axes3[1, 3].set_ylabel("Patches")
    axes3[1, 3].legend(fontsize=8); axes3[1, 3].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path3 = f"{OUT_DIR}/competition_analysis_img{img_idx+1}.png"
    fig3.savefig(out_path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {out_path3}")

    return {
        "sa_final_mse":    sa_loss[-1],
        "gatst_final_mse": gatst_loss[-1],
        "sa_mean_entropy":    sa_ent_vals.mean(),
        "gatst_mean_entropy": gatst_ent_vals.mean(),
        "sa_excl_mean":    sa_excl.mean(),
        "gatst_excl_mean": gatst_excl.mean(),
    }


if __name__ == "__main__":
    all_results = []
    for i, path in enumerate(IMG_PATHS):
        res = make_figure(path, i)
        all_results.append(res)

    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    for i, r in enumerate(all_results):
        print(f"\nImage {i+1}:")
        print(f"  Recon MSE        SA={r['sa_final_mse']:.5f}  GATST={r['gatst_final_mse']:.5f}")
        print(f"  Mean entropy     SA={r['sa_mean_entropy']:.3f}   GATST={r['gatst_mean_entropy']:.3f}")
        print(f"  Patch exclusivity SA={r['sa_excl_mean']:.4f}  GATST={r['gatst_excl_mean']:.4f}")

    print(f"\nAll visualizations saved to: {OUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")
