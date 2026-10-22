import torch
import torch.nn.functional as F

from src.features.extraction import extract_features, extract_dense_features
from src.utils.metrics import compute_gram
from src.utils.random import generate_random_delta
from src.utils.upsampling import upsampling


def fgsm_attack(model, normalisation, x, epsilon, objective="cls"):
    """
    Generate an adversarial example using FGSM, for representation-level.

    The attack maximises the change in the model output representation by applying
    a perturbation based on the gradient of the feature similarity loss.

    Params:
        model: torch.nn.Module
            Vision transformer model.
        x: torch.Tensor
            Preprocessed input image tensor.
        epsilon: float
            Perturbation magnitude.

    Returns:
        torch.Tensor
            Adversarial image tensor.
    """

    x_adv = x.clone().detach()
    x_adv.requires_grad_(True)

    # detach the clean representation so gradients are computed only through x_adv
    with torch.no_grad():
        clean_cls, clean_patch = extract_features(model, normalisation(x))

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

    grad = torch.autograd.grad(loss, x_adv)[0]

    print("Loss:", loss.item())
    print("Grad abs mean:", grad.abs().mean().item())
    print("Grad abs max:", grad.abs().max().item())
    print("Nonzero grad:", (grad != 0).float().mean().item())

    perturbation = epsilon * grad.sign()

    # minimise cosine similarity or maximise mse
    x_adv = x_adv + direction * perturbation

    # keep valid pixel range
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv


def rfgsm_attack(model, normalisation, x, epsilon, seed, objective="cls"):
    """
    Generate an adversarial example using random-start FGSM, for representation-level.

    The attack maximises the change in the model output representation by applying
    a perturbation based on the gradient of the feature similarity loss.

    Params:
        model: torch.nn.Module
            Vision transformer model.
        normalisation:

        x: torch.Tensor
            Preprocessed input image tensor.
        epsilon: float
            Perturbation magnitude.
        objective:


    Returns:
        torch.Tensor
            Adversarial image tensor.
    """

    x_clean = x.clone().detach()

    # random initialisation within the L-inf epsilon ball
    delta = generate_random_delta(x, epsilon, seed)

    x_adv = torch.clamp(x_clean + delta, 0.0, 1.0).detach()
    x_adv.requires_grad_(True)

    # detach the clean representation so gradients are computed only through x_adv
    with torch.no_grad():
        clean_cls, clean_patch = extract_features(model, normalisation(x_clean))

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

    grad = torch.autograd.grad(loss, x_adv)[0]

    perturbation = epsilon * grad.sign()

    # minimise cosine similarity or maximise mse
    x_adv = x_adv + direction * perturbation

    # project back into the L-inf epsilon ball
    delta = torch.clamp(x_adv - x_clean, -epsilon, epsilon)

    # keep valid pixel range
    x_adv = torch.clamp(x_clean + delta, 0.0, 1.0).detach()

    return x_adv


def fgsm_seg_attack(model, head, x, masks, epsilon):
    """
    Generate an adversarial example using FGSM, for segmentation.

    The attack maximises the segmentation cross-entropy loss through the frozen DINO
    backbone and trained segmentation head.

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

    Returns:
        torch.Tensor
            Adversarial image tensor.
    """

    x_adv = x.clone().detach()
    x_adv.requires_grad_(True)

    # gradients pass through the backbone features during attack generation
    features = extract_dense_features(model, x_adv, no_grad=False)

    logits = head(features)
    logits = upsampling(logits, size=masks.shape[-2:])

    loss = F.cross_entropy(logits, masks, ignore_index=255)
    loss.backward()

    perturbation = epsilon * x_adv.grad.sign()

    # maximise cross entropy loss
    x_adv = x_adv + perturbation

    return x_adv.detach()


def fgsm_depth_attack(model, head, x, depths, epsilon):
    """
    Generate an adversarial example using FGSM, for depth estimation.

    The attack maximises the Smooth L1 depth regression loss through the frozen DINO
    backbone and trained depth estimation head.

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

    Returns:
        torch.Tensor
            Adversarial image tensor.
    """

    x_adv = x.clone().detach()
    x_adv.requires_grad_(True)

    # gradients pass through the backbone features during attack generation
    features = extract_dense_features(model, x_adv, no_grad=False)

    pred_depth = head(features)
    pred_depth = upsampling(pred_depth, size=depths.shape[-2:])

    loss = F.smooth_l1_loss(pred_depth, depths)
    loss.backward()

    perturbation = epsilon * x_adv.grad.sign()

    # maximise regression loss
    x_adv = x_adv + perturbation

    return x_adv.detach()
