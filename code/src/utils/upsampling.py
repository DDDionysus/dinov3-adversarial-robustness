import torch.nn.functional as F


def upsampling(logits, size):
    """
    Upsample dense model predictions to the requested spatial resolution using
    bilinear interpolation.
    """

    return F.interpolate(
        logits,
        size=size,
        mode="bilinear",
        align_corners=False
    )
