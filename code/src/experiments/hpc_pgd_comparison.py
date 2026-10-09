from pathlib import Path

import matplotlib.pyplot as plt
import os
import math
import csv
import torch.nn.functional as F
from PIL import Image

import torch

from src.utils.device import get_device
from src.utils.transforms import get_transform
from src.utils.save_as import save_csv
from src.features.extraction import extract_features
from src.attacks.pgd import pgd_attack
from src.models.hpc_loader import load_model
from src.datasets.imagenet import ImageNetValDataset


def evaluate_model(model, image, transform, device, epsilon, alpha, steps):
    """
    Evaluate representation robustness of a model under PGD attack on a single image.

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
            FGSM perturbation strength.

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

    # preprocess image
    x = transform(image).unsqueeze(0).to(device)

    # clean feature
    clean_feature, clean_patch_tokens = extract_features(model, x)

    # generate adversarial image
    x_adv = pgd_attack(model, x, epsilon, alpha, steps)

    # adversarial feature
    adv_feature, adv_patch_tokens = extract_features(model, x_adv)

    cls_similarity = F.cosine_similarity(clean_feature, adv_feature, dim=1).item()

    patch_similarity = F.cosine_similarity(clean_patch_tokens, adv_patch_tokens, dim=1)

    patch_similarity_mean = patch_similarity.mean().item()

    patch_similarity_std = patch_similarity.std().item()

    return {
        "cls_similarity": cls_similarity,
        "patch_similarity_mean": patch_similarity_mean,
        "patch_similarity_std": patch_similarity_std,
    }


def main():

    device = get_device()
    print("Using device:", device)

    project_root = Path("/scratch_tide/hw2825")
    dataset_path = project_root / "datasets/imagenet/val"
    # output_path = project_root / f"outputs/pgd_imagenet_L_subset.csv"

    epsilon = [
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
    ]

    # v2
    dinov2 = load_model("dinov2_vitl14", device)

    transform_v2 = get_transform(resolution=448)

    # v3
    dinov3 = load_model("dinov3_vitl16", device)

    transform_v3 = get_transform(resolution=448)

    dataset = ImageNetValDataset(dataset_path)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    chunk_size = math.ceil(len(dataset) / num_tasks)

    start_idx = task_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(dataset))

    output_path = (
        project_root
        / "pgd_outputs"
        / f"pgd_imagenet_L_chunk_{task_id:02d}.csv"
    )

    print(
        f"Task {task_id}/{num_tasks - 1}: "
        f"Processing images {start_idx} to {end_idx - 1}"
    )

    results = []

    for idx in range(start_idx, end_idx):
    # for idx in range(len(dataset)):
        img, name = dataset[idx]

        print(f"Processing {idx+1}/{len(dataset)}: {name}")

        for e in epsilon:
            alpha = e / 5

            v2_score = evaluate_model(dinov2, img, transform_v2, device, e, alpha, 10)
            v3_score = evaluate_model(dinov3, img, transform_v3, device, e, alpha, 10)

            results.append({
                "image": name,
                "epsilon": e,
                "dinov2_cls": v2_score,
                "dinov3_cls": v3_score,
            })

    save_csv(results, output_path)


if __name__ == "__main__":
    main()
