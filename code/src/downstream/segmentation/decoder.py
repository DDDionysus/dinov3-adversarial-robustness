import torch.nn as nn


class LinearSegmentationHead(nn.Module):
    """
    Lightweight linear prediction head for semantic segmentation.
    A 1x1 convolution maps the backbone feature dimension at each spatial location
    to class logits for the semantic classes without introducing spatial mixing

    Params:
        embed_dim: int
            Feature dimension of the DINO backbone.
        num_classes: int
            Number of classes.
    """

    def __init__(self, embed_dim, num_classes):
        super().__init__()

        self.projection = nn.Conv2d(
            embed_dim,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        """
        Map dense backbone features to per-class segmentation logits.
        """

        return self.projection(x)
