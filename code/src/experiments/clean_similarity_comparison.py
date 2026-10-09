from pathlib import Path
from PIL import Image, ImageOps

import matplotlib.pyplot as plt

from src.features.extraction import extract_features
from src.evaluation.similarity import compute_similarity_map
from src.utils.device import get_device
from src.utils.transforms import get_transform
from src.models.loader import load_model


def plot_comparison(image, similarity_map_v2, similarity_map_v3, output_path):
    """
    Plot the input image and patch-token similarity maps for DINOv2 and DINOv3.
    A shared colour scale is used for both similarity maps so that the colours
    represent the same cosine similarity values and can be compared directly.

    Params:
        image: PIL.Image.Image
            Original image.
        similarity_map_v2: torch.Tensor
            DINOv2 patch-token similarity map.
        similarity_map_v3: torch.Tensor
            DINOv3 patch-token similarity map.
        output_path: str or pathlib.Path
            Path to save the figure.
    """

    vmin = min(
        similarity_map_v2.min().item(),
        similarity_map_v3.min().item(),
    )

    vmax = max(
        similarity_map_v2.max().item(),
        similarity_map_v3.max().item(),
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Input image
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input image")

    # DINOv2 similarity map
    im = axes[1].imshow(
        similarity_map_v2.detach().cpu().numpy(),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].axis("off")
    axes[1].set_title("DINOv2 patch similarity")

    # DINOv3 similarity map
    axes[2].imshow(
        similarity_map_v3.detach().cpu().numpy(),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].axis("off")
    axes[2].set_title("DINOv3 patch similarity")

    char = fig.colorbar(
        im,
        ax=axes[1:],
        label="Cosine similarity",
        fraction=0.025,
        pad=0.02,
    )
    char.set_label("Cosine similarity")

    # plt.tight_layout()

    if output_path:
        plt.savefig(
            output_path,
            bbox_inches="tight",
        )

    plt.show()


def main():
    """
    Generate clean patch-token similarity maps for DINOv2 and DINOv3, using the same
    input image and centre patches.
    """

    transform_resolution = 448

    project_root = Path(__file__).resolve().parent.parent.parent

    image_path = project_root / "input/test_image.jpg"
    # output_path = project_root / f"output/test_output_{transform_resolution}.jpg"

    # check device
    device = get_device()

    # transform - preprocessing
    transform = get_transform(transform_resolution)

    dinov2 = load_model("dinov2_vitl14", device=device)
    dinov3 = load_model("dinov3_vitl16", device=device)

    # load image
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)

    x = transform(image).unsqueeze(0).to(device)

    v2_global_feature, v2_patch_tokens = extract_features(
        dinov2,
        x,
    )

    v3_global_feature, v3_patch_tokens = extract_features(
        dinov3,
        x,
    )

    v2_grid_size = int(v2_patch_tokens.shape[0] ** 0.5)

    print("v2 grid size:", v2_grid_size, "x", v2_grid_size)

    v3_grid_size = int(v3_patch_tokens.shape[0] ** 0.5)

    print("v3 grid size:", v3_grid_size, "x", v3_grid_size)
    # row = grid_size // 2
    # col = grid_size // 2

    v2_row = v2_grid_size // 2
    v2_col = v2_grid_size // 2

    v3_row = v3_grid_size // 2
    v3_col = v3_grid_size // 2

    similarity_map_v2, _ = compute_similarity_map(
        patch_tokens=v2_patch_tokens,
        row=v2_row,
        col=v2_col,
        grid_size=v2_grid_size,
    )

    similarity_map_v3, _ = compute_similarity_map(
        patch_tokens=v3_patch_tokens,
        row=v3_row,
        col=v3_col,
        grid_size=v3_grid_size,
    )

    print("Similarity map shape:", similarity_map_v2.shape)
    print("Similarity map shape:", similarity_map_v3.shape)

    output_path = project_root / f"output/similarity_map/similarity_comparison_res{transform_resolution}.jpg"

    plot_comparison(image, similarity_map_v2, similarity_map_v3, output_path)


if __name__ == "__main__":
    main()
