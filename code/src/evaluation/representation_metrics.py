"""Per-image CLS, corresponding-patch, and Gram robustness metrics.

The metrics use the original conventions in src.utils.metrics:
  - CLS/patch: cosine similarity of corresponding clean/adversarial tokens.
  - Gram: cosine-normalized patch tokens, G = X @ X.T (P x P).
  - Gram MSE: mean of squared G_adv - G_clean over ALL elements.
  - Normalized Gram difference: Frobenius norm of the OFF-DIAGONAL
    difference, divided by the Frobenius norm of the clean off-diagonal.

Only scalar summaries are returned; no token arrays or Gram matrices are
saved. This function is for one image at a time, not a dataset-wide mean.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.metrics import compute_gram


def representation_metrics(
    clean_cls: torch.Tensor,
    clean_patch: torch.Tensor,
    adv_cls: torch.Tensor,
    adv_patch: torch.Tensor,
) -> dict[str, float | int]:
    """Compute robustness metrics from extracted DINO features for B=1.

    Shapes: CLS [1, D_cls], patch [1, P, D_patch] with P >= 2.
    Patch positions MUST correspond (same crop, token order and resolution).
    For 448x448 inputs, the ViT-L/7B patch grids may differ by model;
    metric values describe within-model clean-versus-adversarial changes.

    ``patch_similarity_std`` uses the previous evaluator's torch.std()
    default (sample std, N-1). Percentiles use NumPy's linear interpolation.
    For numerical stability, similarities/Gram metrics are computed in float32.
    No model gradients are needed for metric evaluation.
    """
    pairs = ((clean_cls, adv_cls, 2, "CLS"),
             (clean_patch, adv_patch, 3, "patch"))
    for clean, adv, ndim, name in pairs:
        if not isinstance(clean, torch.Tensor) or not isinstance(adv, torch.Tensor):
            raise TypeError(f"{name} features must be torch tensors")
        if clean.ndim != ndim or adv.shape != clean.shape:
            raise ValueError(f"{name} clean and attacked feature shapes must match ({ndim}D)")
        if clean.shape[0] != 1:
            raise ValueError("Representation metrics require a single image (batch size 1)")
        if clean.device != adv.device:
            raise ValueError(f"{name} feature devices must match")
        if not torch.isfinite(clean).all() or not torch.isfinite(adv).all():
            raise ValueError(f"{name} features contain NaN/Inf")
    if clean_cls.shape[-1] == 0 or clean_patch.shape[-1] == 0:
        raise ValueError("Feature dimension must be nonempty")
    if clean_patch.shape[1] < 2:
        raise ValueError("At least two patch tokens are required for off-diagonal Gram metrics")

    with torch.no_grad():
        # float32 also avoids large fp16 roundoff in similarity / Gram scores.
        cls_cos = F.cosine_similarity(clean_cls.float(), adv_cls.float(), dim=-1)[0]
        patch_cos = F.cosine_similarity(clean_patch.float(), adv_patch.float(), dim=-1)[0]

        # Only P patch similarities are moved to CPU, not P x D embeddings.
        scores = patch_cos.detach().cpu().numpy().astype(np.float64, copy=False)
        p10, p25, p50, p75, p90 = np.percentile(
            scores, [10, 25, 50, 75, 90], method="linear"
        )

        gram_clean = compute_gram(clean_patch)
        gram_adv = compute_gram(adv_patch)
        gram_delta = gram_adv - gram_clean
        gram_mse = gram_delta.square().mean()

        # Match compute_normalised_gram_difference exactly: the diagonal is
        # excluded only for the normalized ratio, not for full Gram MSE.
        p = clean_patch.shape[1]
        off_diag = ~torch.eye(p, dtype=torch.bool, device=gram_clean.device)
        numerator = torch.linalg.vector_norm(gram_delta[:, off_diag], dim=-1)
        denominator = torch.linalg.vector_norm(gram_clean[:, off_diag], dim=-1)
        gram_diff_norm = (numerator / (denominator + 1e-8)).mean()

    return {
        "cls_similarity": float(cls_cos.item()),
        "patch_similarity_mean": float(scores.mean()),
        "patch_similarity_std": float(scores.std(ddof=1)),
        "patch_similarity_min": float(scores.min()),
        "patch_similarity_p10": float(p10),
        "patch_similarity_p25": float(p25),
        "patch_similarity_median": float(p50),
        "patch_similarity_p75": float(p75),
        "patch_similarity_p90": float(p90),
        "patch_similarity_max": float(scores.max()),
        "patch_similarity_count": int(scores.size),
        "gram_mse": float(gram_mse.item()),
        "gram_diff_norm": float(gram_diff_norm.item()),
    }
