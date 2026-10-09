import torch
import torch.nn.functional as F

from src.features.extraction import extract_features, extract_dense_features
from src.utils.metrics import compute_gram
from src.utils.random import generate_random_delta
from src.utils.upsampling import upsampling


def pgd_attack(model, normalisation, x, epsilon, alpha, steps, seed, objective="cls"):
    """
    Generate an adversarial example using PGD, for representation-level.

    The attack iteratively maximises the change in the model output representation.
    A random perturbation is used for initialisation, and the perturbation is
    projected back into the L-infinity epsilon-ball after each update.

    Params:
        model: torch.nn.Module
            Vision transformer model.
        x: torch.Tensor
            Preprocessed input image tensor.
        epsilon: float
            Perturbation magnitude.
        alpha: float
            Step size for each PGD iteration.
        steps: int
            Number of PGD attack iterations.

    Returns:
        torch.Tensor
            Adversarial image tensor.
    """

    x_clean = x.clone().detach()

    # random initial perturbation
    delta = generate_random_delta(x, epsilon, seed)

    x_adv = torch.clamp(x_clean + delta, 0.0, 1.0).detach()

    # detach the clean representation so gradients are computed only through x_adv
    with torch.no_grad():
        clean_cls, clean_patch = extract_features(model, normalisation(x_clean))

    for _ in range(steps):
        x_adv.requires_grad_(True)
        adv_cls, adv_patch = extract_features(model, normalisation(x_adv))

        if objective == "cls":
            loss = F.cosine_similarity(clean_cls, adv_cls, dim=-1).mean()
            direction = -1
        elif objective == "patch":
            loss = F.cosine_similarity(clean_patch, adv_patch, dim=-1).mean()
            direction = -1
        elif objective == "gram":
            clean_gram = compute_gram(clean_patch)
            adv_gram = compute_gram(adv_patch)

            loss = F.mse_loss(clean_gram, adv_gram)
            direction = 1
        else:
            raise ValueError(f"Unknown attack objective: {objective}")

        loss.backward()

        # minimise similarity
        perturbation = alpha * x_adv.grad.sign()

        x_adv = x_adv + direction * perturbation

        # project perturbation back into epsilon range
        delta = torch.clamp(x_adv - x_clean, -epsilon, epsilon)

        x_adv = torch.clamp(x_clean + delta, 0.0, 1.0).detach()

    return x_adv


def pgd_seg_attack(model, head, x, masks, epsilon, alpha, steps):
    """
    Generate an adversarial example using PGD, for segmentation.

    The attack iteratively maximises the segmentation cross-entropy loss.
    A random perturbation is used for initialisation, and the perturbation is
    projected back into the L-infinity epsilon-ball after each update.

    Params:
        model: torch.nn.Module
            Vision transformer model.
        head: torch.nn.Module
            Trained segmentation head.
        x: torch.Tensor
            Preprocessed input image tensor.
        masks: torch.Tensor
            Ground truth segmentation masks.
        epsilon: float
            Perturbation magnitude.
        alpha: float
            Step size for each PGD iteration.
        steps: int
            Number of PGD attack iterations.

    Returns:
        torch.Tensor
            Adversarial image tensor.
    """

    x_clean = x.clone().detach()

    # random initial perturbation
    x_adv = x_clean + torch.empty_like(x_clean).uniform_(-epsilon, epsilon)

    for _ in range(steps):
        x_adv.requires_grad_(True)

        # gradients pass through the backbone features during attack generation
        features = extract_dense_features(model, x_adv, no_grad=False)

        logits = head(features)
        logits = upsampling(logits, size=masks.shape[-2:])

        loss = F.cross_entropy(logits, masks, ignore_index=255)

        model.zero_grad()
        head.zero_grad()

        loss.backward()

        step = alpha * x_adv.grad.sign()

        # maximise cross entropy loss
        x_adv = x_adv + step

        # project perturbation back into epsilon range
        delta = torch.clamp(x_adv - x_clean, -epsilon, epsilon)

        x_adv = x_clean + delta
        x_adv = x_adv.detach()

    return x_adv


def pgd_depth_attack(model, head, x, depths, epsilon, alpha, steps):
    """
    Generate an adversarial example using PGD, for depth estimation.

    The attack iteratively maximises the Smooth L1 depth regression loss.
    A random perturbation is used for initialisation, and the perturbation is
    projected back into the L-infinity epsilon-ball after each update.

    Params:
        model: torch.nn.Module
            Vision transformer model.
        head: torch.nn.Module
            Trained depth estimation head.
        x: torch.Tensor
            Preprocessed input image tensor.
        depths: torch.Tensor
            Ground truth depth maps.
        epsilon: float
            Perturbation magnitude.
        alpha: float
            Step size for each PGD iteration.
        steps: int
            Number of PGD attack iterations.

    Returns:
        torch.Tensor
            Adversarial image tensor.
    """

    x_clean = x.clone().detach()

    # random initial perturbation
    x_adv = x_clean + torch.empty_like(x_clean).uniform_(-epsilon, epsilon)

    for _ in range(steps):
        x_adv.requires_grad_(True)

        features = extract_dense_features(model, x_adv, no_grad=False)

        pred_depth = head(features)
        pred_depth = upsampling(pred_depth, size=depths.shape[-2:])

        loss = F.smooth_l1_loss(pred_depth, depths)

        model.zero_grad()
        head.zero_grad()

        loss.backward()

        step = alpha * x_adv.grad.sign()

        # maximise depth regression loss
        x_adv = x_adv + step

        # project perturbation back into epsilon range
        delta = torch.clamp(x_adv - x_clean, -epsilon, epsilon)

        x_adv = x_clean + delta
        x_adv = x_adv.detach()

    return x_adv
