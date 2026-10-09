import torch.nn.functional as F


def compute_similarity_map(patch_tokens, row, col, grid_size):
    """
    Compute cosine similarity between one reference path and all patches in an image.

    Params:
        patch_tokens: torch.Tensor
            Patch-level feature representations with shape [num_patches, feature_dim].
        row: int
            Row index of the reference patch.
        col: int
            Column index of the reference patch.
        grid_size: int
            Spatial size of the patch grid.

    Returns:
        tuple:
            similarity_map: torch.Tensor
                Cosine similarity map with shape [grid_size, grid_size].
            patch_index: int
                Flattened index of the reference patch.
    """

    patch_index = row * grid_size + col
    selected_patch = patch_tokens[patch_index]

    patch_tokens_norm = F.normalize(patch_tokens, dim=1)
    query_norm = F.normalize(selected_patch, dim=0)

    similarity = patch_tokens_norm @ query_norm
    similarity_map = similarity.reshape(grid_size, grid_size)

    return similarity_map, patch_index
