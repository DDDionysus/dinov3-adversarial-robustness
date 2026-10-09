def get_embed_dim(model_name):
    """
    Get the embedding dimension for a supported DINO model variant.

    Params:
        model_name: str
            DINO model name.

    returns:
        int
            Feature embedding dimension fo the selected backbone.
    """

    embed_dim_map = {
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,

            "dinov3_vitl16": 1024,
            "dinov3_vit7b16": 4096,

            # prototype models
            "dinov2_vits14": 384,
            "dinov3_vits16": 384,
    }

    if model_name not in embed_dim_map:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return embed_dim_map[model_name]
