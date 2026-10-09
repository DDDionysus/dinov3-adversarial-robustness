from pathlib import Path
from PIL import Image

from src.features.extraction import extract_features
from src.evaluation.similarity import compute_similarity_map
from src.utils.device import get_device
from src.utils.transforms import get_transform
from src.models.dinov3 import load_dinov3


def main():
    transform_resolution = 1024

    project_root = Path(__file__).resolve().parent.parent.parent

    checkpoint_path = project_root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    image_path = project_root / "input/test_img.jpg"
    # output_path = project_root / f"output/test_output_{transform_resolution}.jpg"

    # check device
    device = get_device()

    # transform - preprocessing
    transform = get_transform(transform_resolution)

    model = load_dinov3(checkpoint_path, device)

    # load image
    image = Image.open(image_path).convert("RGB")

    x = transform(image).unsqueeze(0).to(device)

    global_feature, patch_tokens = extract_features(
        model,
        x,
    )

    grid_size = int(patch_tokens.shape[0] ** 0.5)
    # num_patches = patch_tokens.shape[0]

    print("Grid size:", grid_size, "x", grid_size)

    # row = grid_size // 2
    # col = grid_size // 2

    positions = [
        (0, 0),
        (grid_size // 2, grid_size // 2),
        (grid_size - 1, grid_size - 1),
        (grid_size // 4, grid_size // 4),
        (3 * grid_size // 4, 3 * grid_size // 4),
    ]

    for row, col in positions:
        similarity_map, patch_index = compute_similarity_map(
            patch_tokens=patch_tokens,
            row=row,
            col=col,
            grid_size=grid_size,
        )

        print(f"Reference patch: row={row}, col={col}, patch_index={patch_index}")
        print("Similarity map shape:", similarity_map.shape)

        output_path = project_root / f"output/similarity_map/similarity_res{transform_resolution}_r{row}_{col}.jpg"

        # query_index = (grid_size // 2) * grid_size + (grid_size // 2)
        # query = patch_tokens[query_index]

        """
        plot_similarity_map(
            image=image,
            similarity_map=similarity_map,
            row=row,
            col=col,
            patch_index=patch_index,
            resolution=transform_resolution,
            output_path=str(output_path),
        )
        """


if __name__ == "__main__":
    main()
