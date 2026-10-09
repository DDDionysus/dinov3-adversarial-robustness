from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


class VOCSegmentationDataset(Dataset):
    """
    Dataset wrapper for Pascal VOC 2012 segmentation.
    This dataset have two modes: In representation mode, it returns the RGB image
    and image identifier; Otherwise, it returns the transformed RGB image with
    segmentation mask.

    Params:
        root: str or pathlib.Path
            Directory of Pascal VOC 2012 dataset.
        split: str
            Dataset split
        transform: callable, optional
            Transform applied to RGB images.
        mask_transform: callable, optional
            Transform applied to segmentation masks.
        representation: bool
            Return images for representation or segmentation.

    Returns:
        representation=True:
            PIL.Image.Image:
                RGB image
            str:
                Image identifier.

        representation=False:
            PIL.Image.Image:
                Transformed RGB image.
            torch.Tensor:
                Segmentation mask.
    """

    def __init__(self, root, split="train", transform=None, mask_transform=None, representation=False):
        self.root = Path(root)
        self.transform = transform
        self.mask_transform = mask_transform
        self.representation = representation

        # read image identifiers from the requested split
        split_file = self.root / f"ImageSets/Segmentation/{split}.txt"

        with open(split_file) as f:
            self.images = [
                line.strip()
                for line in f.readlines()
            ]

        self.image_dir = self.root / "JPEGImages"
        self.mask_dir = self.root / "SegmentationClass"


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        name = self.images[idx]

        image_path = self.image_dir / f"{name}.jpg"
        mask_path = self.mask_dir / f"{name}.png"

        # RGB conversion ensures 3-channel model input
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # representation-level only need RGB image
        if self.representation:
            return image, name

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
