from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


def get_transform(resolution):
    """
    Transform sample without normalisation.

    Params:
        resolution: int
            Resolution of input image

    Returns:
        torchvision.transforms.Compose:
            Image preprocessing transform.
    """

    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])


def get_normalise():
    """
    Return normalisation.
    """

    return transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def get_mask_transform(resolution):
    """
    Create segmentation mask processing pipeline for DINO models.
    Using nearest neighbour interpolation to preserve discrete class labels.

    Params:
        resolution: int
            Resolution of input image

    Returns:
        torchvision.transforms.Compose:
            Mask preprocessing transform.
    """

    return transforms.Compose([
        transforms.Resize(
            resolution,
            interpolation=InterpolationMode.NEAREST
        ),
        transforms.CenterCrop(resolution),
        transforms.PILToTensor()
    ])


def get_depth_transform(resolution):
    """
    Create depth map processing pipeline for DINO models.
    Using bilinear interpolation, since depth values are continuous.

    Params:
        resolution: int
            Resolution of input image

    Returns:
        torchvision.transforms.Compose:
            Depth map preprocessing transform.
    """

    return transforms.Compose([
        transforms.Resize(
            resolution,
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(resolution),
    ])
