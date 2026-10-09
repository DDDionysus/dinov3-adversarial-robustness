import torch.nn as nn


class LinearDepthHead(nn.Module):
    """
    Lightweight linear prediction head for monocular depth estimation.
    A 1x1 convolution maps the backbone feature dimension at each spatial location
    to a single depth value without introducing spatial mixing.

    Params:
        embed_dim: int
            Feature dimension of the DINO backbone.
    """

    def __init__(self, embed_dim):
        super().__init__()

        self.projection = nn.Conv2d(
            embed_dim,
            1,
            kernel_size=1,
        )


    def forward(self, x):
        """
        Map dense backbone features to a single-channel depth prediction.
        """

        return self.projection(x)
