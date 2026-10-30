"""Evaluation of paired clean/adversarial DINO representations (one image)."""

import torch

from src.evaluation.representation_metrics import representation_metrics
from src.features.extraction import extract_features


def evaluate_model(model, device, x_clean, x_adv, normalisation):
    """Return per-image CLS, patch-distribution, and Gram robustness metrics.

    x_clean and x_adv must be identically preprocessed [1,3,H,W] pixel-domain
    tensors. Both pass through the same normalisation and frozen backbone.
    The old result keys are preserved for existing CSV/evaluation consumers.
    """
    if x_clean.shape != x_adv.shape or x_clean.ndim != 4 or x_clean.shape[0] != 1:
        raise ValueError("Expected matching [1, C, H, W] clean and attacked inputs")
    x_clean = x_clean.to(device)
    x_adv = x_adv.to(device)

    with torch.no_grad():
        clean_cls, clean_patch = extract_features(model, normalisation(x_clean))
        adv_cls, adv_patch = extract_features(model, normalisation(x_adv))
        return representation_metrics(clean_cls, clean_patch, adv_cls, adv_patch)
