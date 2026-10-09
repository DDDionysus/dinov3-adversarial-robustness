from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.visualisation.plots import plot_similarity_map
from src.utils.device import get_device
from src.utils.transforms import get_transform
from src.evaluation.similarity import compute_similarity_map
from src.features.extraction import extract_features
from src.models.dinov2 import load_dinov2


def main():
    transform_resolution = 1120

    project_root = Path(__file__).resolve().parent.parent.parent

    # checkpoint_path = project_root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    image_path = project_root / "input/test_img.jpg"

    # get device
    device = get_device()

    # transform - preprocessing
    transform = get_transform(transform_resolution)

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    model = load_dinov2(device)

    global_feature, patch_tokens = extract_features(
        model,
        x,
    )

    num_patches = patch_tokens.shape[0]
    grid_size = int(num_patches ** 0.5)

    assert grid_size * grid_size == num_patches

    row = grid_size // 2
    col = grid_size // 2

    similarity_map, patch_index = compute_similarity_map(
        patch_tokens,
        row,
        col,
        grid_size,
    )

    plot_similarity_map(image, similarity_map)

    print("-----------------")
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Feature keys:", features.keys())

    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)

    print("-----------------")


if __name__ == "__main__":
    main()
