import torch
import torch.nn.functional as F

from src.utils.metrics import compute_gram_mse, compute_normalised_gram_difference
from src.features.extraction import extract_features


def evaluate_model(
        model,
        device,
        x_clean,
        x_adv,
        normalisation,
):
    """
    Evaluate representation robustness of a model under R-FGSM attack on a single image.

    Params:
        model: torch.nn.Module
            Vision transformer model.
        image:
            Input PIL image.
        transform:
            Image preprocessing pipeline.
        device:
            Device to use.
        epsilon:
            R-FGSM perturbation strength.

    Returns:
        dict[str, float]:
            Dictionary containing representation robustness metrics:
            - cls_similarity:
                Cosine similarity between clean and adversarial global features.
            - patch_similarity_mean:
                Mean cosine similarity between corresponding clean and adversarial patch tokens.
            - patch_similarity_std:
                Standard deviation of patch-level cosine similarities.
    """

    x_clean = x_clean.to(device)
    x_adv = x_adv.to(device)

    # clean feature
    with torch.no_grad():
        clean_feature, clean_patch_tokens = extract_features(model, normalisation(x_clean))
        adv_feature, adv_patch_tokens = extract_features(model, normalisation(x_adv))

    cls_similarity = F.cosine_similarity(clean_feature, adv_feature, dim=-1).item()
    patch_similarity = F.cosine_similarity(clean_patch_tokens, adv_patch_tokens, dim=-1)
    patch_similarity_mean = patch_similarity.mean().item()
    patch_similarity_std = patch_similarity.std().item()
    gram_mse = compute_gram_mse(clean_patch_tokens, adv_patch_tokens)
    gram_diff_norm = compute_normalised_gram_difference(clean_patch_tokens, adv_patch_tokens)

    return {
        "cls_similarity": cls_similarity,
        "patch_similarity_mean": patch_similarity_mean,
        "patch_similarity_std": patch_similarity_std,
        "gram_mse": gram_mse,
        "gram_diff_norm": gram_diff_norm,
    }
