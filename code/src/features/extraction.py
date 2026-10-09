import torch


def extract_features(model, x):
    """
    Extract global and patch-level features from a vision transformer.

    Params:
        model: torch.nn.Module
            Vision transformer model.
        x: torch.Tensor
            Preprocessed input image tensor.

    Returns:
        tuple:
            global_features:
                Global image representation.
            patch_tokens:
                Dense patch-level representation.
    """

    features = model.forward_features(x)

    global_feature = features["x_norm_clstoken"]
    patch_tokens = features["x_norm_patchtokens"]

    return global_feature, patch_tokens


def extract_dense_features(model, x, no_grad=True):
    """
    Extract dense patch features for downstream dense prediction tasks.

    Params:
        model:
            DINO model
        x:
            preprocessed image tensor [B,3,H,W]

    Returns:
        patch feature map:
            [B,D,H_patch,W_patch]
    """

    if no_grad:
        with torch.no_grad():
            features = model.forward_features(x)
    else:
        features = model.forward_features(x)

    patch_tokens = features["x_norm_patchtokens"]

    B, N, D = patch_tokens.shape

    grid_size = int(N ** 0.5)

    patch_tokens = patch_tokens.reshape(B, grid_size, grid_size, D)
    patch_tokens = patch_tokens.permute(0, 3, 1, 2)

    return patch_tokens
