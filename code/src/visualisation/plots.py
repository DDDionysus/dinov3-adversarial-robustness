import matplotlib.pyplot as plt
from PIL import Image


def origin_plot_similarity_map(image, similarity_map, row, col, patch_index, resolution, output_path):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image.resize((resolution, resolution)))
    plt.axis("off")
    plt.title("Input image")

    plt.subplot(1, 2, 2)
    plt.imshow(similarity_map.detach().cpu().numpy(), cmap="viridis")
    plt.colorbar()
    plt.axis("off")
    plt.title(f"Patch similarity\nrow={row}, col={col}, index={patch_index}")

    plt.suptitle(f"DINOv3 ViT-L/16, resolution={resolution}")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(str(output_path), bbox_inches="tight")

    plt.show()


def origin_plot_map_2(image, similarity_map, row, col, patch_index, transform_resolution):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image.resize((transform_resolution, transform_resolution)))
    plt.axis("off")
    plt.title("Input image")

    plt.subplot(1, 2, 2)
    plt.imshow(similarity_map.detach().cpu().numpy(), cmap="viridis")
    plt.colorbar()
    plt.axis("off")
    plt.title(
        f"DINOv2 patch similarity\n"
        f"row={row}, col={col}, index={patch_index}"
    )

    plt.tight_layout()
    plt.show()


def plot_similarity_map(image, similarity_map, output_path=None, title="Patch similarity map"):
    """
    Plot input image and patch similarity map.

    Params:
        image: PIL.Image.Image
            Original input image.
        similarity_map: torch.Tensor
            Similarity map between patches.
        output_path: str or Path
            Path to save the figure.
        title: str
            Title of the figure.
    """

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input image")

    plt.subplot(1, 2, 2)
    plt.imshow(
        similarity_map.detach().cpu().numpy(),
        cmap="viridis",
    )
    plt.axis("off")
    plt.title(title)

    plt.tight_layout()

    if output_path:
        plt.savefig(
            output_path,
            bbox_inches="tight",
        )

    plt.show()
